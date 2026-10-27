"""SQL agent: menjawab pertanyaan statistik lowongan dengan query SQLite.

Alur: LLM menulis SQL -> validasi -> pola judul dipotong ke kata dasar -> repository
mengeksekusi query -> LLM merangkum hasil. Data SQL dikirim ke supervisor sebagai tool
artifact; supervisor melengkapi jawaban dengan ``finalize_answer``.
"""

import re
from collections.abc import Iterable
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.agents.prompts import (
    SQL_ANSWER_SYSTEM,
    SQL_GENERATION_SYSTEM,
    SQL_GENERATION_USER,
    SQL_NO_SALARY_NOTE,
    SQL_NOT_RELEVANT_TOKEN,
    SQL_RETRY_USER,
    SQL_TRUNCATED_NOTE,
)
from src.helpers.format import format_salary
from src.models.agent import LLMUsage
from src.repositories.job_repository import JobRepository

DEFAULT_MAX_ROWS = 50
MAX_SQL_ATTEMPTS = 2

# String literal dan komentar dikenali dalam satu pola agar "--" di dalam string tidak
# terbaca sebagai komentar, dan kata seperti 'delete' di dalam string tidak ditolak.
_STRING_OR_COMMENT = re.compile(r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/", re.DOTALL)
_STARTS_WITH_SELECT = re.compile(r"^(select|with)\b", re.IGNORECASE)
# REPLACE hanya ditolak sebagai pernyataan (REPLACE INTO); fungsi replace(...) tetap boleh.
_WRITE_KEYWORD = re.compile(
    r"\b(insert|update|delete|replace\s+into|create|drop|alter|attach|detach|pragma|vacuum"
    r"|reindex)\b",
    re.IGNORECASE,
)
_TRAILING_LIMIT = re.compile(r"\blimit\s+\d+(\s*(,|offset)\s*\d+)?\s*$", re.IGNORECASE)
_TITLE_PATTERN = re.compile(r"(lower\(\s*job_title\s*\)\s+LIKE\s+')([^']*)(')", re.IGNORECASE)
# Akhiran pelaku dipotong ke kata dasar: '%design%' juga mencocokkan "Designer", tetapi
# '%designer%' tidak mencocokkan "Graphic Design" (T11-03 menjawab 1, gold 2).
_TITLE_STEMS = {"designer": "design", "desainer": "desain"}
_TITLE_STEM_WORD = re.compile(r"\b(" + "|".join(_TITLE_STEMS) + r")\b")
_CITATION_TAG = re.compile(r"\[ID:\s*(\d+)\]", re.IGNORECASE)
# Kalimat jumlah lowongan tanpa gaji versi LLM ("12 lowongan lain tidak mencantumkan gaji.",
# "Terdapat 12 lowongan tanpa gaji." di T1b-02); dibuang lalu ditulis ulang dari kolom
# lowongan_tanpa_gaji.
_NO_SALARY_SENTENCE = re.compile(
    r"[ \t]*(?:(?:terdapat|ada)\s+)?\d+\s+lowongan\s+(?:lain\s+)?"
    r"(?:tanpa|tidak\s+mencantumkan)\s+gaji\.?",
    re.IGNORECASE,
)
# "<n> dari <m> lowongan mencantumkan gaji" hanya sah jika hasil punya kolom
# lowongan_dengan_gaji. Di bawah daftar lowongan LLM menulisnya dari angka karangan
# (T3-02: "4 lowongan mencantumkan gaji dari 6 total lowongan").
_SALARY_COUNT_SENTENCE = re.compile(
    r"[ \t]*(?:\S+\s+dari\s+\S+\s+lowongan\s+mencantumkan\s+gaji"
    r"|\d+\s+lowongan\s+mencantumkan\s+gaji\s+dari\s+\d+[^\n.]*)\.?",
    re.IGNORECASE,
)


class SQLQueryResult(BaseModel):
    """Data mentah hasil tool SQL, termasuk ID lowongan untuk supervisor.

    ``usage`` mencatat panggilan LLM penulis SQL di dalam tool. Panggilan ini tidak
    muncul di ``messages`` agent, jadi supervisor menjumlahkannya dari artifact.
    """

    sql: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    job_ids: list[int] = Field(default_factory=list)
    error: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)


class SqlAgent:
    def __init__(self, llm: BaseChatModel, job_repository: JobRepository) -> None:
        self.llm = llm
        self.job_repository = job_repository

    def create_sql_agent(self) -> CompiledStateGraph:
        query_jobs_tool = StructuredTool.from_function(
            self.query_jobs,
            name="query_jobs",
            description="Ambil data lowongan untuk satu pertanyaan statistik atau filter.",
            response_format="content_and_artifact",
        )
        return create_agent(
            model=self.llm,
            system_prompt=SQL_ANSWER_SYSTEM,
            tools=[query_jobs_tool],
            name="sql_worker",
        )

    def query_jobs(self, question: str) -> tuple[str, SQLQueryResult]:
        """Tool agent: buat SQL tervalidasi dan ambil baris dari repository.

        Isi tool untuk LLM adalah JSON hasil tanpa ``usage``; ``usage`` hanya ada di
        artifact untuk supervisor.
        """
        sql: str | None = None
        error: str | None = None
        usage = LLMUsage()

        for _ in range(MAX_SQL_ATTEMPTS):
            try:
                sql = self.generate_sql(question, previous_sql=sql, error=error, usage=usage)
                if sql.upper().startswith(SQL_NOT_RELEVANT_TOKEN):
                    result = SQLQueryResult(error=SQL_NOT_RELEVANT_TOKEN, usage=usage)
                    return result.model_dump_json(exclude={"usage"}), result

                sql = self.stem_title_patterns(self.validate_sql(sql))
                rows = self.job_repository.execute_sql(sql, max_rows=DEFAULT_MAX_ROWS)
                job_ids = [row["id"] for row in rows if isinstance(row.get("id"), int)]
                result = SQLQueryResult(sql=sql, rows=rows, job_ids=job_ids, usage=usage)
                return result.model_dump_json(exclude={"usage"}), result
            except Exception as exc:  # noqa: BLE001 - error SQL dipakai untuk satu kali retry
                error = str(exc)

        result = SQLQueryResult(sql=sql, error=error, usage=usage)
        return result.model_dump_json(exclude={"usage"}), result

    def generate_sql(
        self,
        question: str,
        *,
        previous_sql: str | None,
        error: str | None,
        usage: LLMUsage | None = None,
    ) -> str:
        """Minta LLM menulis satu query; jika ``error`` terisi, pakai prompt retry.

        Panggilan dicatat di ``usage`` dengan langkah ``sql_generate`` atau ``sql_retry``.
        """
        schema = self.job_repository.get_jobs_schema()
        user_prompt = (
            SQL_GENERATION_USER.format(question=question)
            if error is None
            else SQL_RETRY_USER.format(
                question=question, previous_sql=previous_sql or "-", error=error
            )
        )
        response = self.llm.invoke(
            [
                SystemMessage(
                    f"{SQL_GENERATION_SYSTEM.format(max_rows=DEFAULT_MAX_ROWS)}"
                    f"\n\nSkema aktual dari database:\n{schema}"
                ),
                HumanMessage(user_prompt),
            ]
        )
        if usage is not None:
            usage.add(response, step="sql_generate" if error is None else "sql_retry")
        return self.extract_sql(response.text)

    def extract_sql(self, text: str) -> str:
        """Ambil isi blok ```sql jika respons LLM memakai Markdown."""
        text = text.strip()
        if not text.startswith("```"):
            return text

        _, _, sql = text.partition("\n")
        return sql.partition("```")[0].strip()

    def validate_sql(self, sql: str) -> str:
        """Terima hanya satu pernyataan SELECT/WITH yang membaca data, lalu pastikan ada LIMIT.

        Komentar dibuang dari query yang dikembalikan. String literal tidak ikut
        diperiksa, jadi ``LIKE '%delete%'`` tetap sah. Pengaman terakhir tetap
        koneksi ``query_only`` di ``JobRepository.execute_sql``; validasi ini memberi
        pesan error yang jelas untuk retry sebelum query menyentuh database.

        Raises:
            ValueError: Query kosong, lebih dari satu pernyataan, tidak diawali
                SELECT/WITH, atau memuat perintah yang mengubah database.
        """
        cleaned = _STRING_OR_COMMENT.sub(
            lambda match: match.group(0) if match.group(0).startswith("'") else " ", sql
        )
        cleaned = cleaned.strip().rstrip(";").strip()
        code = _STRING_OR_COMMENT.sub("''", cleaned)
        if not cleaned:
            raise ValueError("Query kosong.")
        if ";" in code:
            raise ValueError("Hanya satu pernyataan SQL yang diizinkan.")
        if not _STARTS_WITH_SELECT.match(code):
            raise ValueError("Hanya query SELECT atau WITH yang diizinkan.")
        if _WRITE_KEYWORD.search(code):
            raise ValueError("Query hanya boleh membaca data.")
        if not _TRAILING_LIMIT.search(code):
            cleaned = f"{cleaned}\nLIMIT {DEFAULT_MAX_ROWS}"
        return cleaned

    def stem_title_patterns(self, sql: str) -> str:
        """Potong akhiran pelaku di pola ``lower(job_title) LIKE '...'``: designer -> design.

        Pola kata dasar mencakup pola aslinya, jadi tidak ada lowongan yang hilang;
        tambahannya hanya judul berkata dasar sama ("Graphic Design" untuk "graphic
        designer"). Pola lokasi dan perusahaan tidak diubah. Aturan ejaan di prompt
        tidak cukup: LLM tetap menulis '%graphic designer%' di T11-03.
        """

        def stem(match: re.Match[str]) -> str:
            pattern = _TITLE_STEM_WORD.sub(lambda word: _TITLE_STEMS[word.group(1)], match[2])
            return f"{match[1]}{pattern}{match[3]}"

        return _TITLE_PATTERN.sub(stem, sql)


def format_job_line(row: dict[str, Any]) -> str:
    """Satu baris lowongan dengan format jawaban SQL: ``- [ID:x] Judul - Perusahaan (Lokasi)``.

    Rentang gaji ditambahkan jika baris punya ``salary_min``.
    """
    location = " ".join(str(row.get("location") or "").split())
    line = f"- [ID:{row['id']}] {row['job_title']} - {row.get('company_name') or '-'} ({location})"
    if row.get("salary_min") is not None:
        line += f": {format_salary(row['salary_min'], row.get('salary_max'))}"
    return line


def complete_job_list(answer: str, rows: Iterable[dict[str, Any]]) -> str:
    """Sisipkan baris lowongan hasil query yang tidak dikutip LLM setelah baris ``[ID:x]`` terakhir.

    Hanya berlaku jika jawaban sudah berupa daftar (minimal satu ``[ID:x]``); jawaban
    statistik tidak diubah. Aturan "tampilkan setiap baris" di prompt tidak cukup: LLM
    menyaring ulang dan membuang lowongan bergaji 9-12 juta untuk "di atas 10 juta"
    (T1b-02) dan 7-10 juta untuk "di atas 8 juta" (T3-02).
    """
    cited = {int(job_id) for job_id in _CITATION_TAG.findall(answer)}
    if not cited:
        return answer
    missing = [
        format_job_line(row)
        for row in rows
        if isinstance(row.get("id"), int) and row["id"] not in cited and row.get("job_title")
    ]
    if not missing:
        return answer
    lines = answer.split("\n")
    last = max(index for index, line in enumerate(lines) if _CITATION_TAG.search(line))
    return "\n".join([*lines[: last + 1], *missing, *lines[last + 1 :]])


def finalize_answer(
    answer: str, rows: list[dict[str, Any]], max_rows: int = DEFAULT_MAX_ROWS
) -> str:
    """Tulis dengan kode bagian jawaban yang faktanya sudah pasti di hasil query.

    Urutan: daftar dilengkapi (``complete_job_list``); kalimat "<n> dari <m> lowongan
    mencantumkan gaji" dibuang jika hasil tidak punya kolom ``lowongan_dengan_gaji``;
    kalimat jumlah lowongan tanpa gaji versi LLM diganti ``SQL_NO_SALARY_NOTE`` dari
    kolom ``lowongan_tanpa_gaji`` (hanya jika > 0); catatan ``SQL_TRUNCATED_NOTE`` jika
    jumlah baris mencapai ``max_rows``. Tanpa baris hasil, jawaban tidak diubah.
    """
    if not rows:
        return answer
    answer = complete_job_list(answer, rows)
    if "lowongan_dengan_gaji" not in rows[0]:
        answer = _SALARY_COUNT_SENTENCE.sub("", answer)
    no_salary = rows[0].get("lowongan_tanpa_gaji")
    if no_salary is not None:
        answer = _NO_SALARY_SENTENCE.sub("", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    if isinstance(no_salary, int) and no_salary > 0:
        answer = f"{answer}\n\n{SQL_NO_SALARY_NOTE.format(count=no_salary)}"
    if len(rows) >= max_rows:
        answer = f"{answer}\n\n{SQL_TRUNCATED_NOTE.format(max_rows=max_rows)}"
    return answer
