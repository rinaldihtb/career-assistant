"""SQL agent: menjawab pertanyaan statistik lowongan dengan query SQLite.

Alur: LLM menulis SQL -> validasi -> repository mengeksekusi query -> LLM
merangkum hasil. Data SQL dikirim ke supervisor sebagai tool artifact.
"""

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
from src.repositories.job_repository import JobRepository

DEFAULT_MAX_ROWS = 50
MAX_SQL_ATTEMPTS = 2


class SQLQueryResult(BaseModel):
    """Data mentah hasil tool SQL, termasuk ID lowongan untuk supervisor."""

    sql: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    job_ids: list[int] = Field(default_factory=list)
    error: str | None = None


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
        """Tool agent: buat SQL tervalidasi dan ambil baris dari repository."""
        sql: str | None = None
        error: str | None = None

        for _ in range(MAX_SQL_ATTEMPTS):
            try:
                sql = self.generate_sql(question, previous_sql=sql, error=error)
                if sql.upper().startswith(SQL_NOT_RELEVANT_TOKEN):
                    result = SQLQueryResult(error=SQL_NOT_RELEVANT_TOKEN)
                    return result.model_dump_json(), result

                sql = self.validate_sql(sql)
                rows = self.job_repository.execute_sql(sql)
                job_ids = [row["id"] for row in rows if isinstance(row.get("id"), int)]
                result = SQLQueryResult(sql=sql, rows=rows, job_ids=job_ids)
                return result.model_dump_json(), result
            except Exception as exc:  # noqa: BLE001 - error SQL dipakai untuk satu kali retry
                error = str(exc)

        result = SQLQueryResult(sql=sql, error=error)
        return result.model_dump_json(), result

    def generate_sql(self, question: str, *, previous_sql: str | None, error: str | None) -> str:
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
        return self.extract_sql(response.text)

    def extract_sql(self, text: str) -> str:
        """Ambil isi blok ```sql jika respons LLM memakai Markdown."""
        text = text.strip()
        if not text.startswith("```"):
            return text

        _, _, sql = text.partition("\n")
        return sql.partition("```")[0].strip()

    def validate_sql(self, sql: str) -> str:
        """Pastikan query hanya baca data dan batasi jumlah hasilnya."""
        cleaned = sql.strip().removesuffix(";").strip()
        first_word = cleaned.split(maxsplit=1)[0].lower() if cleaned else ""
        if first_word not in {"select", "with"}:
            raise ValueError("Hanya query SELECT atau WITH yang diizinkan.")

        if "limit" not in cleaned.lower().split():
            cleaned = f"{cleaned}\nLIMIT {DEFAULT_MAX_ROWS}"
        return cleaned
