"""SQL agent lengkap dengan LLM palsu (gratis, tanpa API key)."""

import re
import sqlite3
from contextlib import closing

from agents.prompts import (
    SQL_ANSWER_SYSTEM,
    SQL_EMPTY_RESULT_MESSAGE,
    SQL_FAILED_MESSAGE,
    SQL_GENERATION_SYSTEM,
    SQL_NO_SALARY_NOTE,
    SQL_NOT_RELEVANT_MESSAGE,
    SQL_NOT_RELEVANT_TOKEN,
    SQL_TRUNCATED_NOTE,
)
from agents.sql_agent import execute_sql, run_sql_agent, strip_invalid_citations, validate_sql


def _sql_block(sql: str) -> str:
    return f"```sql\n{sql}\n```"


def test_success_uses_two_llm_calls(jobs_db, fake_llm):
    llm = fake_llm(
        _sql_block("SELECT COUNT(*) AS jumlah FROM jobs WHERE lower(location) LIKE '%jakarta%'"),
        "Ada 3 lowongan di Jakarta.",
    )
    result = run_sql_agent("Berapa lowongan di Jakarta?", llm=llm, db_path=jobs_db)

    assert result.answer == "Ada 3 lowongan di Jakarta."
    assert result.rows == [{"jumlah": 3}]
    assert result.sql.endswith("LIMIT 50")
    assert result.error is None
    assert result.usage.llm_calls == 2
    assert result.usage.total_tokens == 30
    assert [call.step for call in result.usage.calls] == ["sql_generate", "sql_answer"]


def test_answer_prompt_contains_query_result(jobs_db, fake_llm):
    llm = fake_llm(_sql_block("SELECT COUNT(*) AS jumlah FROM jobs"), "Ada 6 lowongan.")
    run_sql_agent("Total lowongan?", llm=llm, db_path=jobs_db)

    answer_prompt = llm.received[1][-1].content
    assert '{"jumlah": 6}' in answer_prompt


def test_returns_job_ids_from_rows(jobs_db, fake_llm):
    llm = fake_llm(
        _sql_block("SELECT id, job_title FROM jobs WHERE lower(job_title) LIKE '%data analyst%'"),
        "Ada 2 lowongan [ID:1] [ID:2].",
    )
    result = run_sql_agent("Lowongan data analyst?", llm=llm, db_path=jobs_db)
    assert result.job_ids == [1, 2]


def test_title_stem_is_applied_before_execution(jobs_db, fake_llm):
    llm = fake_llm(
        _sql_block("SELECT id FROM jobs WHERE lower(job_title) LIKE '%graphic designer%'"), "ok"
    )
    result = run_sql_agent("Lowongan graphic designer", llm=llm, db_path=jobs_db)

    assert "LIKE '%graphic design%'" in result.sql
    assert result.rows == [{"id": 3}]
    assert result.sql in llm.received[1][-1].content


def test_retry_once_after_sql_error(jobs_db, fake_llm):
    llm = fake_llm(
        _sql_block("SELECT kolom_palsu FROM jobs"),
        _sql_block("SELECT COUNT(*) AS jumlah FROM jobs"),
        "Ada 6 lowongan.",
    )
    result = run_sql_agent("Total lowongan?", llm=llm, db_path=jobs_db)

    assert result.answer == "Ada 6 lowongan."
    assert result.error is None
    assert result.usage.llm_calls == 3
    retry_prompt = llm.received[1][-1].content
    assert "kolom_palsu" in retry_prompt
    assert "no such column" in retry_prompt


def test_gives_up_after_two_failures(jobs_db, fake_llm):
    llm = fake_llm(
        _sql_block("SELECT kolom_palsu FROM jobs"),
        _sql_block("SELECT kolom_lain FROM jobs"),
        "TIDAK BOLEH DIPANGGIL",
    )
    result = run_sql_agent("Total lowongan?", llm=llm, db_path=jobs_db)

    assert result.answer == SQL_FAILED_MESSAGE
    assert "no such column" in result.error
    assert result.usage.llm_calls == 2


def test_llm_refusal_text_triggers_retry(jobs_db, fake_llm):
    llm = fake_llm(
        "Maaf, saya tidak bisa membuat query.",
        _sql_block("SELECT COUNT(*) AS jumlah FROM jobs"),
        "Ada 6 lowongan.",
    )
    result = run_sql_agent("Total lowongan?", llm=llm, db_path=jobs_db)
    assert result.answer == "Ada 6 lowongan."
    assert result.usage.llm_calls == 3


def test_write_query_from_llm_never_changes_data(jobs_db, fake_llm):
    llm = fake_llm(
        _sql_block("DELETE FROM jobs"),
        _sql_block("WITH x AS (SELECT 1) DELETE FROM jobs"),
    )
    result = run_sql_agent("Hapus semua lowongan", llm=llm, db_path=jobs_db)

    assert result.answer == SQL_FAILED_MESSAGE
    with closing(sqlite3.connect(jobs_db)) as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 6


def test_empty_result_skips_answer_llm(jobs_db, fake_llm):
    llm = fake_llm(
        _sql_block("SELECT id FROM jobs WHERE lower(location) LIKE '%papua%'"),
        "TIDAK BOLEH DIPANGGIL",
    )
    result = run_sql_agent("Lowongan di Papua?", llm=llm, db_path=jobs_db)

    assert result.answer == SQL_EMPTY_RESULT_MESSAGE
    assert result.usage.llm_calls == 1


def test_llm_error_returns_fixed_message(jobs_db, fake_llm):
    llm = fake_llm(RuntimeError("API down"))
    result = run_sql_agent("Total lowongan?", llm=llm, db_path=jobs_db)

    assert result.answer == SQL_FAILED_MESSAGE
    assert "API down" in result.error
    assert result.usage.llm_calls == 0


def test_not_relevant_question_stops_after_one_call(jobs_db, fake_llm):
    llm = fake_llm(SQL_NOT_RELEVANT_TOKEN, "TIDAK BOLEH DIPANGGIL")
    result = run_sql_agent("Hapus semua data lowongan", llm=llm, db_path=jobs_db)

    assert result.answer == SQL_NOT_RELEVANT_MESSAGE
    assert result.error is None
    assert result.usage.llm_calls == 1


def test_not_relevant_token_is_written_in_prompt():
    assert SQL_NOT_RELEVANT_TOKEN in SQL_GENERATION_SYSTEM


def test_long_text_is_truncated_only_in_prompt(jobs_db, fake_llm):
    # hex(zeroblob(300)) menghasilkan teks 600 karakter "0".
    llm = fake_llm(_sql_block("SELECT id, hex(zeroblob(300)) AS teks FROM jobs WHERE id = 1"), "ok")
    result = run_sql_agent("Tampilkan teks panjang", llm=llm, db_path=jobs_db)

    answer_prompt = llm.received[1][-1].content
    assert "0" * 201 not in answer_prompt
    assert len(result.rows[0]["teks"]) == 600


def test_truncated_note_is_added_by_code_only_when_rows_hit_limit(jobs_db, fake_llm):
    llm = fake_llm(_sql_block("SELECT id FROM jobs"), "Daftar lowongan.")
    result = run_sql_agent("Semua lowongan", llm=llm, db_path=jobs_db, max_rows=2)

    assert result.answer == "Daftar lowongan.\n\n" + SQL_TRUNCATED_NOTE.format(max_rows=2)
    assert "terpotong" not in llm.received[1][-1].content


def test_no_truncated_note_when_rows_below_limit(jobs_db, fake_llm):
    llm = fake_llm(_sql_block("SELECT id FROM jobs WHERE id = 1"), "Satu lowongan.")
    result = run_sql_agent("Lowongan ID 1", llm=llm, db_path=jobs_db)

    assert result.answer == "Satu lowongan."


_SALARY_LIST_SQL = (
    "SELECT id, job_title, company_name, location, salary_min, salary_max, "
    "(SELECT COUNT(*) FROM jobs WHERE salary_min IS NULL) AS lowongan_tanpa_gaji "
    "FROM jobs WHERE salary_max > 9000000 ORDER BY salary_max DESC"
)


def test_rows_missing_from_answer_are_appended_by_code(jobs_db, fake_llm):
    # Hasil: ID 4 (15-20 juta) dan ID 1 (8-10 juta); LLM membuang ID 1.
    llm = fake_llm(
        _sql_block(_SALARY_LIST_SQL),
        "Lowongan bergaji di atas 9 juta:\n- [ID:4] Marketing Manager - PT Delta (Jakarta Raya)",
    )
    result = run_sql_agent("Lowongan gaji di atas 9 juta", llm=llm, db_path=jobs_db)

    assert result.answer.splitlines()[2] == (
        "- [ID:1] Data Analyst - PT Alpha (Jakarta Selatan, Jakarta Raya): "
        "Rp8.000.000 - Rp10.000.000"
    )


def test_no_salary_sentence_comes_from_column_not_llm(jobs_db, fake_llm):
    llm = fake_llm(
        _sql_block(_SALARY_LIST_SQL),
        "- [ID:4] A\n- [ID:1] B\n\n1 lowongan lain tidak mencantumkan gaji.",
    )
    result = run_sql_agent("Lowongan gaji di atas 9 juta", llm=llm, db_path=jobs_db)

    assert result.answer == "- [ID:4] A\n- [ID:1] B\n\n" + SQL_NO_SALARY_NOTE.format(count=2)


def test_answer_without_list_or_zero_no_salary_is_unchanged(jobs_db, fake_llm):
    llm = fake_llm(
        _sql_block(
            "SELECT id, job_title, (SELECT COUNT(*) FROM jobs WHERE id = 0) AS lowongan_tanpa_gaji "
            "FROM jobs WHERE id IN (1, 3)"
        ),
        "Ada dua lowongan.",
    )
    result = run_sql_agent("Lowongan ID 1 dan 3", llm=llm, db_path=jobs_db)

    assert result.answer == "Ada dua lowongan."


# --- Guard sitasi dan contoh di prompt ---------------------------------------


def test_citation_not_in_query_result_is_removed(jobs_db, fake_llm):
    llm = fake_llm(
        _sql_block("SELECT salary_min, salary_max FROM jobs WHERE id = 1"),
        "- [ID:123] Data Analyst - PT Alpha (Jakarta): Rp8.000.000 - Rp10.000.000",
    )
    result = run_sql_agent("Gaji lowongan dengan ID 1?", llm=llm, db_path=jobs_db)

    assert result.answer == "- Data Analyst - PT Alpha (Jakarta): Rp8.000.000 - Rp10.000.000"
    assert result.job_ids == []
    assert result.error == "ID di luar hasil query dibuang: [123]"


def test_strip_invalid_citations_keeps_valid_ids():
    answer = "- [ID:1] A\n- [ID: 9] B\n- [ID:2] C\n[ID:9] lagi"
    cleaned, invalid = strip_invalid_citations(answer, [1, 2])

    assert cleaned == "- [ID:1] A\n- B\n- [ID:2] C\nlagi"
    assert invalid == [9]


def test_answer_prompt_has_no_numeric_example_id():
    assert not re.search(r"\[ID:\s*\d+\]", SQL_ANSWER_SYSTEM)


def test_generation_prompt_examples_run_on_database(jobs_db):
    """Contoh query di prompt harus valid: LLM meniru contoh apa adanya."""
    examples = re.findall(r"```sql\n(.*?)\n```", SQL_GENERATION_SYSTEM, re.DOTALL)

    assert len(examples) == 7
    for sql in examples:
        execute_sql(validate_sql(sql), jobs_db)
