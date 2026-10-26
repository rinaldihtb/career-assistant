"""SqlAgent dengan LLM palsu (gratis, tanpa API key): tool ``query_jobs`` dan agent lengkap."""

import re
import sqlite3
from contextlib import closing

from conftest import ScriptedLLM, agent_script
from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.prompts import (
    SQL_ANSWER_SYSTEM,
    SQL_GENERATION_SYSTEM,
    SQL_NOT_RELEVANT_TOKEN,
)
from src.agents.sql_agent import DEFAULT_MAX_ROWS, SqlAgent

COUNT_JAKARTA = "SELECT COUNT(*) AS jumlah FROM jobs WHERE lower(location) LIKE '%jakarta%'"


def _count_jobs(db_path) -> int:
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


# --- Tool query_jobs ---------------------------------------------------------


def test_query_jobs_success_uses_one_llm_call(make_repository, fake_llm):
    llm = fake_llm(COUNT_JAKARTA)
    content, result = SqlAgent(llm, make_repository()).query_jobs("Berapa lowongan di Jakarta?")

    assert result.sql == f"{COUNT_JAKARTA}\nLIMIT 50"
    assert result.rows == [{"jumlah": 3}]
    assert result.error is None
    assert content == result.model_dump_json(exclude={"usage"})
    assert len(llm.received) == 1


def test_generation_calls_are_recorded_in_artifact_usage(make_repository, fake_llm):
    llm = fake_llm("SELECT kolom_palsu FROM jobs", "SELECT COUNT(*) AS jumlah FROM jobs")
    content, result = SqlAgent(llm, make_repository()).query_jobs("Total lowongan?")

    assert [call.step for call in result.usage.calls] == ["sql_generate", "sql_retry"]
    assert result.usage.total_tokens == 30
    assert "usage" not in content


def test_generation_prompt_has_rules_and_actual_schema(make_repository, fake_llm):
    llm = fake_llm(COUNT_JAKARTA)
    SqlAgent(llm, make_repository()).query_jobs("Berapa lowongan di Jakarta?")

    system, user = llm.received[0]
    assert system.content.startswith(SQL_GENERATION_SYSTEM.format(max_rows=DEFAULT_MAX_ROWS))
    assert "CREATE TABLE jobs" in system.content.split("Skema aktual dari database:")[1]
    assert user.content == "Pertanyaan: Berapa lowongan di Jakarta?"


def test_markdown_fence_is_removed_before_execution(make_repository, fake_llm):
    llm = fake_llm(f"```sql\n{COUNT_JAKARTA}\n```")
    _, result = SqlAgent(llm, make_repository()).query_jobs("Berapa lowongan di Jakarta?")
    assert result.rows == [{"jumlah": 3}]


def test_returns_job_ids_from_rows(make_repository, fake_llm):
    llm = fake_llm("SELECT id, job_title FROM jobs WHERE lower(job_title) LIKE '%data analyst%'")
    _, result = SqlAgent(llm, make_repository()).query_jobs("Lowongan data analyst?")
    assert result.job_ids == [1, 2]


def test_rows_are_capped_at_max_rows(jobs_db, make_repository, fake_llm):
    llm = fake_llm("SELECT a.id FROM jobs a, jobs b, jobs c LIMIT 1000")
    _, result = SqlAgent(llm, make_repository()).query_jobs("Semua kombinasi")
    assert len(result.rows) == DEFAULT_MAX_ROWS


def test_retry_once_after_sql_error(make_repository, fake_llm):
    llm = fake_llm("SELECT kolom_palsu FROM jobs", "SELECT COUNT(*) AS jumlah FROM jobs")
    _, result = SqlAgent(llm, make_repository()).query_jobs("Total lowongan?")

    assert result.rows == [{"jumlah": 6}]
    assert result.error is None
    retry_prompt = llm.received[1][-1].content
    assert "SELECT kolom_palsu FROM jobs" in retry_prompt
    assert "no such column" in retry_prompt


def test_gives_up_after_two_failures(make_repository, fake_llm):
    llm = fake_llm("SELECT kolom_palsu FROM jobs", "SELECT kolom_lain FROM jobs")
    _, result = SqlAgent(llm, make_repository()).query_jobs("Total lowongan?")

    assert result.rows == []
    assert "no such column" in result.error
    assert len(llm.received) == 2


def test_llm_refusal_text_triggers_retry(make_repository, fake_llm):
    llm = fake_llm("Maaf, saya tidak bisa.", "SELECT COUNT(*) AS jumlah FROM jobs")
    _, result = SqlAgent(llm, make_repository()).query_jobs("Total lowongan?")

    assert result.rows == [{"jumlah": 6}]
    assert "SELECT atau WITH" in llm.received[1][-1].content


def test_write_query_from_llm_never_changes_data(jobs_db, make_repository, fake_llm):
    llm = fake_llm("DELETE FROM jobs", "WITH x AS (SELECT 1) DELETE FROM jobs")
    _, result = SqlAgent(llm, make_repository()).query_jobs("Hapus semua lowongan")

    assert result.rows == [] and result.error is not None
    assert _count_jobs(jobs_db) == 6


def test_not_relevant_question_stops_after_one_call(make_repository, fake_llm):
    llm = fake_llm(SQL_NOT_RELEVANT_TOKEN, "TIDAK BOLEH DIPANGGIL")
    _, result = SqlAgent(llm, make_repository()).query_jobs("Hapus semua data lowongan")

    assert result.error == SQL_NOT_RELEVANT_TOKEN
    assert result.sql is None and result.rows == []
    assert result.usage.llm_calls == 1


def test_not_relevant_token_is_written_in_prompt():
    assert SQL_NOT_RELEVANT_TOKEN in SQL_GENERATION_SYSTEM


# --- Agent lengkap (create_agent) --------------------------------------------


def test_agent_calls_tool_then_answers_from_its_result(make_repository):
    question = "Berapa lowongan di Jakarta?"
    llm = ScriptedLLM(
        reply=agent_script(sql={question: COUNT_JAKARTA}, answers={question: "Ada 3 lowongan."})
    )
    state = (
        SqlAgent(llm, make_repository())
        .create_sql_agent()
        .invoke({"messages": [HumanMessage(question)]})
    )

    assert state["messages"][-1].content == "Ada 3 lowongan."
    tool_message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert tool_message.name == "query_jobs"
    assert tool_message.artifact.rows == [{"jumlah": 3}]
    assert [messages[0].content == SQL_ANSWER_SYSTEM for messages in llm.received] == [
        True,
        False,
        True,
    ]


# --- Prompt ------------------------------------------------------------------


def test_answer_prompt_has_no_numeric_example_id():
    assert not re.search(r"\[ID:\s*\d+\]", SQL_ANSWER_SYSTEM)


def test_generation_prompt_examples_run_on_database(make_repository):
    """Contoh query di prompt harus valid: LLM meniru contoh apa adanya."""
    examples = re.findall(r"```sql\n(.*?)\n```", SQL_GENERATION_SYSTEM, re.DOTALL)
    agent = SqlAgent(llm=None, job_repository=make_repository())

    assert len(examples) == 7
    for sql in examples:
        agent.job_repository.execute_sql(agent.validate_sql(sql), max_rows=DEFAULT_MAX_ROWS)
