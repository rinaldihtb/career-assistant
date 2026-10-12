"""Uji manual SQL agent terhadap data/database.sqlite asli.

    make sample-sql       validasi + eksekusi saja (tanpa LLM, gratis)
    make sample-sql-llm   agent lengkap dengan LLM sungguhan (butuh .env)

Mode LLM mencatat waktu, jumlah panggilan LLM, dan token per pertanyaan,
lalu membandingkan jawaban dengan hasil query yang dihitung langsung.
"""

import os
import sqlite3
import sys
import time
from contextlib import closing

from dotenv import load_dotenv

from agents.prompts import SQL_FAILED_MESSAGE, SQL_NOT_RELEVANT_MESSAGE
from agents.sql_agent import SQLExecutionError, execute_sql, run_sql_agent, validate_sql

load_dotenv()
DB_PATH = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")

# (judul, query) — query sengaja tanpa LIMIT untuk membuktikan LIMIT ditambahkan otomatis.
CHECK_QUERIES = [
    ("Total lowongan", "SELECT COUNT(*) AS total FROM jobs"),
    (
        "Lowongan per tipe kerja",
        "SELECT work_type, COUNT(*) AS jumlah FROM jobs GROUP BY work_type ORDER BY jumlah DESC",
    ),
    (
        "Statistik gaji (yang tidak NULL)",
        (
            "SELECT COUNT(salary_min) AS n, MIN(salary_min) AS min_gaji, "
            "ROUND(AVG(salary_min)) AS rata_gaji_min, MAX(salary_max) AS max_gaji FROM jobs"
        ),
    ),
    (
        "Lowongan di Jakarta",
        "SELECT COUNT(*) AS jumlah FROM jobs WHERE lower(location) LIKE '%jakarta%'",
    ),
    (
        "5 lowongan data analyst",
        (
            "SELECT id, job_title, company_name, location FROM jobs "
            "WHERE lower(job_title) LIKE '%data analyst%' ORDER BY id"
        ),
    ),
]

# Dieksekusi TANPA validate_sql agar yang diuji benar-benar koneksi read-only.
BLOCKED_QUERIES = [
    "WITH x AS (SELECT 1) DELETE FROM jobs",
    "DELETE FROM jobs",
    "SELECT 1; DROP TABLE jobs",
]

# (pertanyaan, query pembanding yang ditulis manual; None = harus ditolak/gagal)
LLM_QUESTIONS = [
    ("Berapa total lowongan yang ada?", "SELECT COUNT(*) FROM jobs"),
    (
        "Berapa jumlah lowongan di Jakarta?",
        "SELECT COUNT(*) FROM jobs WHERE lower(location) LIKE '%jakarta%'",
    ),
    (
        "Ada berapa lowongan dengan tipe kontrak?",
        "SELECT COUNT(*) FROM jobs WHERE work_type = 'Kontrak/Temporer'",
    ),
    (
        "Berapa rata-rata gaji minimum dari lowongan yang mencantumkan gaji?",
        "SELECT ROUND(AVG(salary_min)), COUNT(salary_min) FROM jobs",
    ),
    (
        "Perusahaan apa saja yang paling banyak membuka lowongan? Sebutkan 5 teratas.",
        (
            "SELECT company_name, COUNT(*) AS n FROM jobs "
            "GROUP BY company_name ORDER BY n DESC LIMIT 5"
        ),
    ),
    (
        "Berapa lowongan Data Analyst di Surabaya?",
        (
            "SELECT COUNT(*) FROM jobs WHERE lower(job_title) LIKE '%data analyst%' "
            "AND lower(location) LIKE '%surabaya%'"
        ),
    ),
    (
        "Lowongan apa yang gajinya paling tinggi?",
        (
            "SELECT id, job_title, company_name, salary_max FROM jobs "
            "WHERE salary_max IS NOT NULL ORDER BY salary_max DESC LIMIT 3"
        ),
    ),
    ("Hapus semua data lowongan di database.", None),
]


def _count_rows_independently() -> int:
    """Hitung total baris dengan koneksi biasa sebagai pembanding independen."""
    with closing(sqlite3.connect(DB_PATH)) as db:
        return db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


def _ground_truth(sql: str) -> list[tuple]:
    """Jalankan query pembanding dengan koneksi biasa (bukan lewat agent)."""
    with closing(sqlite3.connect(DB_PATH)) as db:
        return db.execute(sql).fetchall()


def _number_found(answer: str, expected: list[tuple]) -> str:
    """Cek kasar: apakah angka pertama hasil pembanding muncul di jawaban."""
    if len(expected) != 1 or not isinstance(expected[0][0], int | float):
        return "cek manual"
    value = int(expected[0][0])
    candidates = {str(value), f"{value:,}".replace(",", ".")}
    return "COCOK" if any(c in answer for c in candidates) else "TIDAK COCOK"


def run_tools_check() -> None:
    """Query uji tanpa LLM dan bukti bahwa query tulis ditolak."""
    print(f"Database: {DB_PATH}\n")

    for title, raw_sql in CHECK_QUERIES:
        sql = validate_sql(raw_sql, max_rows=5)
        rows = execute_sql(sql, DB_PATH, max_rows=5)
        print(f"== {title}\n{sql}")
        for row in rows:
            print("  ", row)
        print()

    print("== Query yang harus DITOLAK (langsung ke koneksi read-only)")
    before = _count_rows_independently()
    for raw_sql in BLOCKED_QUERIES:
        try:
            execute_sql(raw_sql, DB_PATH)
            print(f"  BAHAYA, lolos: {raw_sql}")
        except SQLExecutionError as error:
            print(f"  ditolak ({error}): {raw_sql!r}")
    after = _count_rows_independently()
    status = "AMAN" if before == after else "BERUBAH!"
    print(f"\nJumlah baris sebelum/sesudah: {before}/{after} -> {status}")


def run_llm_check() -> None:
    """Jalankan agent lengkap dengan LLM sungguhan dan catat metriknya."""
    from config.llm import LLM_MODEL  # import di sini: mode tanpa LLM tidak butuh API key

    summary = []
    for question, truth_sql in LLM_QUESTIONS:
        start = time.perf_counter()
        result = run_sql_agent(question, llm=LLM_MODEL, db_path=DB_PATH)
        elapsed = time.perf_counter() - start

        if truth_sql is None:
            expected = None
            refused = result.answer in (SQL_FAILED_MESSAGE, SQL_NOT_RELEVANT_MESSAGE)
            check = "OK (ditolak)" if refused else "TIDAK DITOLAK"
        else:
            expected = _ground_truth(truth_sql)
            check = _number_found(result.answer, expected)
        summary.append((question, elapsed, result, check))

        print("=" * 100)
        print(f"Q: {question}")
        print(f"SQL agent : {result.sql}")
        print(f"Pembanding: {expected}")
        print(f"Jawaban   : {result.answer}")
        print(f"Error     : {result.error}")
        print(
            f"Waktu {elapsed:.2f}s | LLM calls {result.usage.llm_calls} | "
            f"token in/out/total {result.usage.input_tokens}/{result.usage.output_tokens}/"
            f"{result.usage.total_tokens} | cek angka: {check}"
        )

    print("\n" + "=" * 100)
    print(f"{'Pertanyaan':<55} {'detik':>6} {'calls':>5} {'token':>6}  cek")
    for question, elapsed, result, check in summary:
        print(
            f"{question[:55]:<55} {elapsed:>6.2f} {result.usage.llm_calls:>5} "
            f"{result.usage.total_tokens:>6}  {check}"
        )
    total_tokens = sum(r.usage.total_tokens for _, _, r, _ in summary)
    avg_time = sum(e for _, e, _, _ in summary) / len(summary)
    print(f"\nRata-rata waktu {avg_time:.2f}s | total token {total_tokens}")
    print(f"Jumlah baris setelah semua uji: {_count_rows_independently()} (harus 473)")


if __name__ == "__main__":
    if "--llm" in sys.argv:
        run_llm_check()
    else:
        run_tools_check()
