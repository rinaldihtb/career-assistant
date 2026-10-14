"""SQL agent lengkap dengan LLM palsu (gratis, tanpa API key)."""

import sqlite3
from contextlib import closing

from agents.prompts import (
    SQL_EMPTY_RESULT_MESSAGE,
    SQL_FAILED_MESSAGE,
    SQL_GENERATION_SYSTEM,
    SQL_NOT_RELEVANT_MESSAGE,
    SQL_NOT_RELEVANT_TOKEN,
)
from agents.sql_agent import run_sql_agent


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
