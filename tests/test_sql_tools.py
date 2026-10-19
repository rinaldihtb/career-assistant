"""SQL agent bagian tanpa LLM: validasi, eksekusi, koneksi read-only."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from agents.prompts import SQL_SCHEMA_COLUMNS
from agents.sql_agent import (
    SQLExecutionError,
    UnsafeSQLError,
    execute_sql,
    extract_sql,
    stem_title_patterns,
    validate_sql,
)
from repositories.job_repository import connect_readonly

REAL_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "database.sqlite"


def _count_jobs(db_path) -> int:
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


# --- aturan pencocokan deterministik -----------------------------------------


def test_title_patterns_are_stemmed():
    sql = (
        "WHERE (lower(job_title) LIKE '%graphic designer%' OR lower(job_title) LIKE '%desainer grafis%')"
        " AND lower(location) LIKE '%bandung%'"
    )
    assert stem_title_patterns(sql) == (
        "WHERE (lower(job_title) LIKE '%graphic design%' OR lower(job_title) LIKE '%desain grafis%')"
        " AND lower(location) LIKE '%bandung%'"
    )


def test_title_stem_keeps_other_patterns_and_columns():
    sql = (
        "WHERE lower(job_title) LIKE '%data analyst%' AND lower(company_name) LIKE '%designer%'"
        " AND lower(job_title) LIKE '%design%'"
    )
    assert stem_title_patterns(sql) == sql


# --- extract_sql / validate_sql --------------------------------------------


def test_extract_sql_from_code_fence():
    text = "Berikut query:\n```sql\nSELECT 1\n```\nSemoga membantu."
    assert extract_sql(text) == "SELECT 1"


def test_validate_adds_limit_when_missing():
    assert validate_sql("SELECT * FROM jobs", max_rows=20).endswith("LIMIT 20")


def test_validate_keeps_existing_limit():
    sql = "SELECT * FROM jobs LIMIT 5"
    assert validate_sql(sql) == sql


def test_validate_keeps_limit_with_offset():
    sql = "SELECT * FROM jobs LIMIT 5 OFFSET 10"
    assert validate_sql(sql) == sql


def test_validate_strips_semicolon_fence_and_comments():
    raw = "```sql\n-- hitung semua\nSELECT COUNT(*) FROM jobs LIMIT 1; -- selesai\n```"
    assert validate_sql(raw) == "SELECT COUNT(*) FROM jobs LIMIT 1"


def test_validate_accepts_with_clause():
    sql = "WITH t AS (SELECT * FROM jobs) SELECT COUNT(*) FROM t LIMIT 1"
    assert validate_sql(sql) == sql


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "DELETE FROM jobs",
        "DROP TABLE jobs",
        "UPDATE jobs SET status = 'CLOSED'",
        "PRAGMA table_info(jobs)",
        "ATTACH DATABASE 'x.db' AS x",
        "Maaf, saya tidak bisa menjawab itu.",
    ],
)
def test_validate_rejects_non_select(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


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


# --- execute_sql -----------------------------------------------------------


def test_execute_returns_dict_rows(jobs_db):
    rows = execute_sql(
        validate_sql("SELECT COUNT(*) AS jumlah FROM jobs WHERE lower(location) LIKE '%jakarta%'"),
        jobs_db,
    )
    assert rows == [{"jumlah": 3}]


def test_execute_salary_aggregate_ignores_null(jobs_db):
    rows = execute_sql(
        "SELECT AVG(salary_min) AS rata, COUNT(salary_min) AS n FROM jobs LIMIT 1", jobs_db
    )
    assert rows == [{"rata": 8_750_000.0, "n": 4}]


def test_execute_caps_rows_even_with_large_limit(jobs_db):
    rows = execute_sql("SELECT id FROM jobs LIMIT 1000", jobs_db, max_rows=2)
    assert len(rows) == 2


def test_execute_blocks_write_hidden_in_cte(jobs_db):
    # Diawali WITH sehingga lolos cek bentuk; harus ditolak koneksi read-only.
    # Sengaja tanpa validate_sql: di SQLite bawaan macOS, "DELETE ... LIMIT"
    # sudah gagal sebagai syntax error sebelum mode read-only sempat diuji.
    with pytest.raises(SQLExecutionError, match="readonly"):
        execute_sql("WITH x AS (SELECT 1) DELETE FROM jobs", jobs_db)
    assert _count_jobs(jobs_db) == 6


def test_execute_blocks_multiple_statements(jobs_db):
    with pytest.raises(SQLExecutionError):
        execute_sql("SELECT 1; DROP TABLE jobs", jobs_db)
    assert _count_jobs(jobs_db) == 6


def test_execute_reports_syntax_error(jobs_db):
    with pytest.raises(SQLExecutionError, match="no such column"):
        execute_sql("SELECT kolom_palsu FROM jobs LIMIT 1", jobs_db)


def test_execute_stops_long_running_query(jobs_db):
    endless = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT COUNT(*) FROM c LIMIT 1"
    )
    with pytest.raises(SQLExecutionError, match="batas waktu"):
        execute_sql(endless, jobs_db, timeout_seconds=0.2)


# --- sinkronisasi prompt dengan database asli ------------------------------


@pytest.mark.skipif(not REAL_DB_PATH.is_file(), reason="data/database.sqlite belum di-seed")
def test_schema_prompt_matches_database():
    with closing(connect_readonly(REAL_DB_PATH)) as conn:
        cursor = conn.execute("SELECT * FROM jobs LIMIT 0")
        db_columns = {column[0] for column in cursor.description}
    missing = set(SQL_SCHEMA_COLUMNS) - db_columns
    assert not missing, f"Kolom di prompt tidak ada di database: {missing}"
