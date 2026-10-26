"""SQL agent: menjawab pertanyaan statistik lowongan dengan query SQLite.

Alur: LLM menulis SQL -> validasi -> repository mengeksekusi query -> LLM
merangkum hasil. Data SQL dikirim ke supervisor sebagai tool artifact.
"""

import re
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
    SQL_NOT_RELEVANT_TOKEN,
    SQL_RETRY_USER,
)
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

                sql = self.validate_sql(sql)
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
