"""SQL agent bagian tanpa LLM: validasi query dan eksekusi baca-saja di repository."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from sqlalchemy.exc import DBAPIError

from src.agents.prompts import SQL_SCHEMA_COLUMNS
from src.agents.sql_agent import SqlAgent
from src.repositories.job_repository import connect_readonly

REAL_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "database.sqlite"


@pytest.fixture
def agent() -> SqlAgent:
    return SqlAgent(llm=None, job_repository=None)


def _count_jobs(db_path) -> int:
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


# --- extract_sql / validate_sql --------------------------------------------


def test_extract_sql_from_code_fence(agent):
    assert agent.extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"


def test_validate_adds_limit_when_missing(agent):
    assert agent.validate_sql("SELECT * FROM jobs").endswith("\nLIMIT 50")


def test_validate_keeps_existing_limit(agent):
    sql = "SELECT * FROM jobs LIMIT 5"
    assert agent.validate_sql(sql) == sql


def test_validate_keeps_limit_with_offset(agent):
    sql = "SELECT * FROM jobs LIMIT 5 OFFSET 10"
    assert agent.validate_sql(sql) == sql


def test_validate_adds_limit_when_only_subquery_has_limit(agent):
    sql = "SELECT * FROM (SELECT id FROM jobs LIMIT 10)"
    assert agent.validate_sql(sql) == f"{sql}\nLIMIT 50"


def test_validate_strips_semicolon_and_comments(agent):
    raw = "-- hitung semua\nSELECT COUNT(*) FROM jobs LIMIT 1; -- selesai"
    assert agent.validate_sql(raw) == "SELECT COUNT(*) FROM jobs LIMIT 1"


def test_validate_accepts_with_clause(agent):
    sql = "WITH t AS (SELECT * FROM jobs) SELECT COUNT(*) FROM t LIMIT 1"
    assert agent.validate_sql(sql) == sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM jobs WHERE lower(job_description) LIKE '%delete%' LIMIT 5",
        "SELECT replace(job_title, 'Sr.', 'Senior') AS judul FROM jobs LIMIT 5",
        "SELECT id FROM jobs WHERE job_title LIKE '%;%' LIMIT 5",
        "SELECT id FROM jobs WHERE job_title LIKE '%--%' LIMIT 5",
        "SELECT updated_at FROM jobs LIMIT 5",
    ],
)
def test_validate_ignores_keywords_inside_strings_and_names(agent, sql):
    assert agent.validate_sql(sql) == sql


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "-- hanya komentar",
        "DELETE FROM jobs",
        "DROP TABLE jobs",
        "UPDATE jobs SET status = 'CLOSED'",
        "PRAGMA table_info(jobs)",
        "ATTACH DATABASE 'x.db' AS x",
        "Maaf, saya tidak bisa menjawab itu.",
        "WITH x AS (SELECT 1) DELETE FROM jobs",
        "WITH x AS (SELECT 1) REPLACE INTO jobs (id, job_title) VALUES (1, 'x')",
        "WITH x AS (SELECT 1) INSERT INTO jobs (job_title) SELECT 'x' FROM x",
        "SELECT 1; DROP TABLE jobs",
        "SELECT 1; SELECT 2",
    ],
)
def test_validate_rejects_anything_but_one_read_statement(agent, sql):
    with pytest.raises(ValueError):
        agent.validate_sql(sql)


# --- connect_readonly ------------------------------------------------------


def test_connect_readonly_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        connect_readonly(tmp_path / "tidak_ada.sqlite")


def test_connect_readonly_blocks_direct_write(jobs_db):
    with (
        closing(connect_readonly(jobs_db)) as conn,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        conn.execute("DELETE FROM jobs")
    assert _count_jobs(jobs_db) == 6


# --- JobRepository.execute_sql ---------------------------------------------


def test_execute_returns_dict_rows(make_repository):
    rows = make_repository().execute_sql(
        "SELECT COUNT(*) AS jumlah FROM jobs WHERE lower(location) LIKE '%jakarta%'", max_rows=50
    )
    assert rows == [{"jumlah": 3}]


def test_execute_salary_aggregate_ignores_null(make_repository):
    rows = make_repository().execute_sql(
        "SELECT AVG(salary_min) AS rata, COUNT(salary_min) AS n FROM jobs LIMIT 1", max_rows=50
    )
    assert rows == [{"rata": 8_750_000.0, "n": 4}]


def test_execute_caps_rows_even_with_large_limit(make_repository):
    assert len(make_repository().execute_sql("SELECT id FROM jobs LIMIT 1000", max_rows=2)) == 2


def test_execute_keeps_colon_text_without_bind_parameters(make_repository):
    rows = make_repository().execute_sql("SELECT '10:30' AS jam, ':id' AS teks", max_rows=1)
    assert rows == [{"jam": "10:30", "teks": ":id"}]


@pytest.mark.parametrize(
    "sql",
    [
        "WITH x AS (SELECT 1) DELETE FROM jobs",
        "DELETE FROM jobs",
        "UPDATE jobs SET status = 'CLOSED'",
        "DROP TABLE jobs",
    ],
)
def test_execute_rejects_write_even_without_validation(jobs_db, make_repository, sql):
    with pytest.raises(DBAPIError, match="readonly"):
        make_repository().execute_sql(sql, max_rows=50)
    assert _count_jobs(jobs_db) == 6


def test_execute_rejects_multiple_statements(jobs_db, make_repository):
    with pytest.raises(DBAPIError):
        make_repository().execute_sql("SELECT 1; DROP TABLE jobs", max_rows=50)
    assert _count_jobs(jobs_db) == 6


def test_read_only_mode_does_not_leak_to_next_query(make_repository):
    repository = make_repository()
    with pytest.raises(DBAPIError):
        repository.execute_sql("SELECT kolom_palsu FROM jobs", max_rows=1)
    with repository.db._engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA query_only").scalar() == 0


def test_execute_reports_syntax_error(make_repository):
    with pytest.raises(DBAPIError, match="no such column"):
        make_repository().execute_sql("SELECT kolom_palsu FROM jobs LIMIT 1", max_rows=1)


# --- sinkronisasi prompt dengan database asli ------------------------------


@pytest.mark.skipif(not REAL_DB_PATH.is_file(), reason="data/database.sqlite belum di-seed")
def test_schema_prompt_matches_database():
    with closing(connect_readonly(REAL_DB_PATH)) as conn:
        cursor = conn.execute("SELECT * FROM jobs LIMIT 0")
        db_columns = {column[0] for column in cursor.description}
    missing = set(SQL_SCHEMA_COLUMNS) - db_columns
    assert not missing, f"Kolom di prompt tidak ada di database: {missing}"
