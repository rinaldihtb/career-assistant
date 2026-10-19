"""SQL agent: menjawab pertanyaan statistik lowongan dengan query SQLite.

Alur: LLM menulis SQL -> validasi -> eksekusi read-only -> (retry 1x jika gagal)
-> LLM merangkum hasil -> guard sitasi (tanpa LLM). Jumlah panggilan LLM selalu 1 sampai 3.
"""

import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.prompts import (
    SQL_ANSWER_SYSTEM,
    SQL_ANSWER_USER,
    SQL_EMPTY_RESULT_MESSAGE,
    SQL_FAILED_MESSAGE,
    SQL_GENERATION_SYSTEM,
    SQL_GENERATION_USER,
    SQL_NOT_RELEVANT_MESSAGE,
    SQL_NOT_RELEVANT_TOKEN,
    SQL_RETRY_USER,
)
from models.agent import AgentResult, LLMUsage
from repositories.job_repository import connect_readonly

DEFAULT_MAX_ROWS = 50
DEFAULT_TIMEOUT_SECONDS = 3.0
MAX_SQL_ATTEMPTS = 2  # 1 percobaan + maksimal 1 retry
MAX_CELL_CHARS = 200  # teks panjang (mis. job_description) dipotong sebelum dikirim ke LLM

_CODE_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STARTS_WITH_SELECT = re.compile(r"^(select|with)\b", re.IGNORECASE)
_TRAILING_LIMIT = re.compile(r"\blimit\s+\d+(\s*(,|offset)\s*\d+)?\s*$", re.IGNORECASE)
_CITATION_TAG = re.compile(r"\[ID:\s*(\d+)\]\s*", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    """Query ditolak sebelum dieksekusi (kosong atau bukan SELECT)."""


class SQLExecutionError(RuntimeError):
    """Query gagal saat dieksekusi SQLite (sintaks, kolom salah, timeout, dll.)."""


class LLMCallError(RuntimeError):
    """Panggilan ke provider LLM gagal (jaringan, rate limit, API key)."""


# --- Entry point -------------------------------------------------------------


def run_sql_agent(
    question: str,
    *,
    llm: BaseChatModel,
    db_path: str | Path,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> AgentResult:
    """Jawab pertanyaan statistik lowongan.

    Jumlah panggilan LLM:
        - sukses langsung: 2 (generate + rangkum)
        - sukses setelah retry: 3
        - gagal dua kali: 2, jawaban pesan tetap
        - hasil query kosong: 1, jawaban pesan tetap
        - pertanyaan tidak relevan / minta ubah data: 1, jawaban pesan tetap

    Args:
        question: Pertanyaan pengguna.
        llm: Chat model yang disuntikkan (asli atau palsu saat test).
        db_path: Path ke file SQLite.
        max_rows: Batas jumlah baris yang diambil dan dikirim ke LLM.

    Returns:
        ``AgentResult`` dengan ``source="sql"``. Fungsi ini tidak pernah
        melempar exception; kegagalan dicatat di ``error``.
    """
    usage = LLMUsage()
    try:
        sql, rows, error = _generate_and_execute(question, llm, db_path, max_rows, usage)
        if error == SQL_NOT_RELEVANT_TOKEN:
            return AgentResult(answer=SQL_NOT_RELEVANT_MESSAGE, source="sql", usage=usage)
        if rows is None:
            return AgentResult(
                answer=SQL_FAILED_MESSAGE, source="sql", sql=sql, usage=usage, error=error
            )
        if not rows:
            return AgentResult(answer=SQL_EMPTY_RESULT_MESSAGE, source="sql", sql=sql, usage=usage)

        answer = _call_llm(
            llm,
            SQL_ANSWER_SYSTEM,
            SQL_ANSWER_USER.format(
                question=question,
                sql=sql,
                row_count=len(rows),
                truncated_note=", mungkin terpotong" if len(rows) >= max_rows else "",
                rows=_format_rows_for_prompt(rows),
            ),
            usage,
            step="sql_answer",
        )
    except LLMCallError as exc:
        return AgentResult(answer=SQL_FAILED_MESSAGE, source="sql", usage=usage, error=str(exc))

    job_ids = [r["id"] for r in rows if isinstance(r.get("id"), int)]
    answer, invalid_ids = strip_invalid_citations(answer, job_ids)
    return AgentResult(
        answer=answer,
        source="sql",
        job_ids=job_ids,
        sql=sql,
        rows=rows,
        usage=usage,
        error=f"ID di luar hasil query dibuang: {invalid_ids}" if invalid_ids else None,
    )


# --- Langkah dengan LLM ------------------------------------------------------


def _generate_and_execute(
    question: str, llm: BaseChatModel, db_path: str | Path, max_rows: int, usage: LLMUsage
) -> tuple[str | None, list[dict[str, Any]] | None, str | None]:
    """Generate SQL lalu eksekusi; jika gagal, retry sekali dengan pesan error.

    Returns:
        ``(sql, rows, error)``. ``rows`` bernilai ``None`` jika semua percobaan gagal.
        Jika LLM menilai pertanyaan tidak relevan, ``error`` berisi
        ``SQL_NOT_RELEVANT_TOKEN`` dan tidak ada retry.
    """
    sql: str | None = None
    error: str | None = None
    for _ in range(MAX_SQL_ATTEMPTS):
        sql = generate_sql(question, llm, usage, max_rows=max_rows, previous_sql=sql, error=error)
        if sql.upper().startswith(SQL_NOT_RELEVANT_TOKEN):
            return None, None, SQL_NOT_RELEVANT_TOKEN
        try:
            sql = validate_sql(sql, max_rows)
            return sql, execute_sql(sql, db_path, max_rows), None
        except (UnsafeSQLError, SQLExecutionError) as exc:
            error = str(exc)
    return sql, None, error


def generate_sql(
    question: str,
    llm: BaseChatModel,
    usage: LLMUsage,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    previous_sql: str | None = None,
    error: str | None = None,
) -> str:
    """Minta LLM menulis satu query SQL; jika ada ``error``, gunakan prompt retry.

    Returns:
        Query yang sudah dikeluarkan dari blok ```sql``` tetapi BELUM divalidasi.
    """
    if error is None:
        user_prompt = SQL_GENERATION_USER.format(question=question)
    else:
        user_prompt = SQL_RETRY_USER.format(
            question=question, previous_sql=previous_sql or "-", error=error
        )
    system_prompt = SQL_GENERATION_SYSTEM.format(max_rows=max_rows)
    step = "sql_generate" if error is None else "sql_retry"
    return extract_sql(_call_llm(llm, system_prompt, user_prompt, usage, step=step))


def _call_llm(llm: BaseChatModel, system: str, user: str, usage: LLMUsage, *, step: str) -> str:
    """Satu panggilan LLM: catat token per langkah dan seragamkan semua error provider."""
    try:
        response = llm.invoke([SystemMessage(system), HumanMessage(user)])
    except Exception as exc:  # tipe error provider LLM sangat beragam
        raise LLMCallError(f"LLM error: {exc}") from exc
    usage.add(response, step=step)
    return response.text


# --- Langkah tanpa LLM -------------------------------------------------------


def _format_rows_for_prompt(rows: list[dict[str, Any]]) -> str:
    """Ubah hasil query jadi teks untuk prompt; teks panjang dipotong agar token tetap kecil.

    ``AgentResult.rows`` tetap menyimpan data utuh untuk UI; yang dipotong
    hanya salinan yang dikirim ke LLM.
    """
    lines = []
    for row in rows:
        short = {
            key: f"{value[:MAX_CELL_CHARS]}..."
            if isinstance(value, str) and len(value) > MAX_CELL_CHARS
            else value
            for key, value in row.items()
        }
        lines.append(json.dumps(short, ensure_ascii=False, default=str))
    return "\n".join(lines)


def strip_invalid_citations(answer: str, valid_ids: list[int]) -> tuple[str, list[int]]:
    """Buang tag ``[ID:x]`` yang ID-nya tidak ada di hasil query; teks lainnya tetap.

    Kenapa hanya tag yang dibuang, bukan barisnya: pada baseline eval, baris dengan
    ID palsu tetap memuat fakta yang benar (misalnya gaji), hanya ID-nya yang
    dikarang. Tanpa tag, UI tidak membuat kartu untuk lowongan yang salah.

    Returns:
        ``(jawaban_bersih, id_yang_dibuang)``; daftar ID urut kemunculan, tanpa duplikat.
    """
    valid = set(valid_ids)
    invalid: list[int] = []

    def _check(match: re.Match[str]) -> str:
        job_id = int(match.group(1))
        if job_id in valid:
            return match.group(0)
        invalid.append(job_id)
        return ""

    cleaned = _CITATION_TAG.sub(_check, answer)
    return cleaned, list(dict.fromkeys(invalid))


def extract_sql(text: str) -> str:
    """Ambil isi blok ```sql``` dari balasan LLM; jika tidak ada, kembalikan teks apa adanya."""
    match = _CODE_FENCE.search(text)
    return (match.group(1) if match else text).strip()


def validate_sql(sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> str:
    """Pastikan query berbentuk SELECT dan diakhiri LIMIT.

    Ini pemeriksaan bentuk agar error cepat dan jelas untuk retry. Pengaman
    data yang sesungguhnya adalah koneksi read-only di ``connect_readonly``.

    Raises:
        UnsafeSQLError: Jika query kosong atau tidak diawali SELECT/WITH.
    """
    # Komentar dibuang agar tidak mengganggu deteksi LIMIT di akhir query.
    cleaned = _COMMENT.sub(" ", extract_sql(sql)).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQLError("Query kosong.")
    if not _STARTS_WITH_SELECT.match(cleaned):
        raise UnsafeSQLError("Hanya query SELECT (boleh diawali WITH) yang diizinkan.")
    if not _TRAILING_LIMIT.search(cleaned):
        cleaned = f"{cleaned}\nLIMIT {max_rows}"
    return cleaned


def execute_sql(
    sql: str,
    db_path: str | Path,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Eksekusi query pada koneksi read-only.

    Args:
        sql: Query hasil ``validate_sql``.
        db_path: Path ke file SQLite.
        max_rows: Batas keras jumlah baris, berlaku walaupun LIMIT di query lebih besar.
        timeout_seconds: Query dihentikan jika lebih lama dari ini (mis. JOIN tak sengaja).

    Returns:
        Daftar baris sebagai dict ``{nama_kolom: nilai}``.

    Raises:
        SQLExecutionError: Untuk semua error SQLite. Pesannya dipakai sebagai
            umpan balik saat retry.
    """
    deadline = time.monotonic() + timeout_seconds
    try:
        with closing(connect_readonly(db_path)) as conn:
            # Dicek SQLite tiap ~10.000 instruksi; True = hentikan query.
            conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
            cursor = conn.execute(sql)
            columns = [column[0] for column in cursor.description or []]
            rows = cursor.fetchmany(max_rows)
    # sqlite3.Warning ikut ditangkap: di Python < 3.12 query multi-statement
    # memunculkan Warning, yang bukan turunan sqlite3.Error.
    except (sqlite3.Error, sqlite3.Warning) as exc:
        if time.monotonic() > deadline:
            raise SQLExecutionError(f"Query melebihi batas waktu {timeout_seconds} detik.") from exc
        raise SQLExecutionError(str(exc)) from exc
    return [dict(zip(columns, row)) for row in rows]
