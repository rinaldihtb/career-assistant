"""Fixture bersama untuk semua test.

Test memakai database kecil buatan sendiri, bukan data/database.sqlite,
supaya hasilnya pasti (tidak berubah saat data di-seed ulang) dan cepat.
"""

import itertools
import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from src.agents.prompts import (
    RAG_ANSWER_SYSTEM,
    SQL_ANSWER_SYSTEM,
    SQL_GENERATION_SYSTEM,
    SUPERVISOR_PLANNER_SYSTEM,
)
from src.repositories.job_repository import JobRepository

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

# Disalin dari scripts/seed_database.py agar fixture punya skema yang sama.
JOBS_DDL = """
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    job_title VARCHAR(255) NOT NULL,
    company_name VARCHAR(255),
    location VARCHAR(255),
    work_type VARCHAR(50),
    job_description TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_period VARCHAR(20),
    salary_currency VARCHAR(3),
    status VARCHAR(30),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

SAMPLE_JOBS = [
    (1, "Data Analyst", "PT Alpha", "Jakarta Selatan, Jakarta Raya", "Full time",
     "Analisis data penjualan dengan SQL dan Python.", 8_000_000, 10_000_000, "month", "IDR", "OPEN"),
    (2, "Data Analyst Intern", "PT Beta", "Surabaya, Jawa Timur", "Paruh waktu",
     "Membantu cleaning data dan membuat dashboard.", None, None, None, None, "OPEN"),
    (3, "Graphic Designer", "PT Gamma", "Bandung, Jawa Barat", "Kontrak/Temporer",
     "Membuat materi visual untuk media sosial.", 5_000_000, 6_000_000, "month", "IDR", "OPEN"),
    (4, "Marketing Manager", "PT Delta", "Jakarta Raya\n(Hibrid)", "Full time",
     "Memimpin tim marketing digital.", 15_000_000, 20_000_000, "month", "IDR", "OPEN"),
    (5, "Backend Engineer", "PT Epsilon", "Tangerang, Banten", "Full time",
     "Membangun REST API dengan Python.", None, None, None, None, "OPEN"),
    (6, "HR Generalist", "PT Zeta", "Jakarta Barat, Jakarta Raya", "Full time",
     "Rekrutmen dan administrasi karyawan.", 7_000_000, 7_000_000, "month", "IDR", "OPEN"),
]  # fmt: skip


@pytest.fixture
def jobs_db(tmp_path: Path) -> Path:
    """Buat database SQLite sementara berisi 6 lowongan contoh."""
    db_path = tmp_path / "jobs.sqlite"
    with closing(sqlite3.connect(db_path)) as db:
        db.execute(JOBS_DDL)
        db.executemany(
            """
            INSERT INTO jobs (id, job_title, company_name, location, work_type,
                job_description, salary_min, salary_max, salary_period,
                salary_currency, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            SAMPLE_JOBS,
        )
        db.commit()
    return db_path


class RecordingFakeChatModel(GenericFakeChatModel):
    """LLM palsu yang membalas berurutan dan mencatat semua pesan yang diterima.

    Dipakai di test supaya bisa memeriksa isi prompt (misalnya error masuk
    ke prompt retry) tanpa API key dan tanpa biaya.
    """

    received: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def make_fake_llm(*replies: str | Exception) -> RecordingFakeChatModel:
    """Buat LLM palsu dari daftar balasan teks.

    Setiap balasan diberi ``usage_metadata`` 10 input + 5 output token
    agar penghitungan token juga ikut dites. Jika elemen berupa
    Exception, panggilan tersebut akan melempar exception itu.
    """

    def _replies():
        for reply in replies:
            if isinstance(reply, Exception):
                raise reply
            yield AIMessage(
                content=reply,
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

    return RecordingFakeChatModel(messages=_replies())


@pytest.fixture
def fake_llm():
    """Fixture pabrik LLM palsu: ``fake_llm("balasan 1", "balasan 2", ...)``."""
    return make_fake_llm


class FakeVectorStore:
    """Stub vector store dengan antarmuka ``similarity_search_with_score`` milik Qdrant.

    Tidak memakai embedding: skor setiap chunk ditentukan langsung oleh test.
    Filter ``MatchAny`` pada ``metadata.job_id`` diterapkan seperti Qdrant,
    dan setiap pemanggilan dicatat di ``calls`` untuk diperiksa test.
    """

    def __init__(self, chunks: list[tuple[int, str, float]]):
        self.chunks = chunks
        self.calls: list[dict] = []

    def similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
        self.calls.append({"query": query, "k": k, "filter": filter})
        allowed = None
        if filter is not None:
            allowed = set(filter.must[0].match.any)
        hits = [
            (Document(page_content=text, metadata={"job_id": job_id}), score)
            for job_id, text, score in self.chunks
            if allowed is None or job_id in allowed
        ]
        hits.sort(key=lambda hit: hit[1], reverse=True)
        return hits[:k]


@pytest.fixture
def make_repository(jobs_db):
    """Pabrik ``JobRepository`` di atas ``jobs_db``: ``make_repository([(job_id, teks, skor)])``."""

    def _make(chunks=()) -> JobRepository:
        return JobRepository(
            db=SQLDatabase.from_uri(f"sqlite:///{jobs_db}"), vector_db=FakeVectorStore(list(chunks))
        )

    return _make


# --- LLM palsu untuk agent berbasis create_agent dan supervisor --------------

_TOOL_CALL_IDS = itertools.count(1)


class ScriptedLLM(BaseChatModel):
    """LLM palsu yang memilih balasan dari isi pesan, bukan dari urutan panggilan.

    Sub-tugas supervisor berjalan paralel, jadi urutan panggilan antar worker tidak
    pasti. ``reply`` menerima daftar pesan dan mengembalikan teks, ``AIMessage``
    (misalnya tool call), atau Exception yang lalu dilempar. Setiap panggilan dicatat
    di ``received``. Teks dibalas dengan ``usage_metadata`` 10 input + 5 output token.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reply: Callable[[list[BaseMessage]], Any]
    received: list[list[BaseMessage]] = Field(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        with self._lock:
            self.received.append(messages)
            reply = self.reply(messages)
        if isinstance(reply, Exception):
            raise reply
        message = reply if isinstance(reply, AIMessage) else AIMessage(reply, usage_metadata=USAGE)
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, **args: Any) -> AIMessage:
    """Balasan LLM berupa satu tool call ``name(**args)`` dengan ID unik."""
    call = {"name": name, "args": args, "id": f"call-{next(_TOOL_CALL_IDS)}"}
    return AIMessage(content="", tool_calls=[call], usage_metadata=USAGE)


def plan_reply(*subtasks, response="answer", skipped=(), declined=(), schema="Plan") -> AIMessage:
    """Balasan planner: tool call ``Plan`` (``CVPlan`` saat CV aktif).

    ``subtasks`` berupa pasangan (agent, pertanyaan).
    """
    return tool_call(
        schema,
        response=response,
        subtasks=[{"agent": agent, "question": question} for agent, question in subtasks],
        skipped=list(skipped),
        declined=list(declined),
    )


def agent_script(
    *,
    plan: Any = None,
    sql: dict[str, Any] | None = None,
    answers: dict[str, Any] | None = None,
    chat: Any = None,
) -> Callable[[list[BaseMessage]], Any]:
    """Fungsi ``reply`` untuk ``ScriptedLLM`` yang mengenali setiap jenis prompt.

    Args:
        plan: Balasan planner (``plan_reply(...)`` atau Exception).
        sql: ``{pertanyaan: query}``; daftar query dipakai berurutan untuk retry.
        answers: ``{pertanyaan: jawaban akhir worker}`` setelah tool selesai.
        chat: Balasan prompt lain (percakapan umum dan klarifikasi).

    Panggilan pertama worker RAG/SQL dibalas tool call ``retrieve_jobs``/``query_jobs``
    dengan pertanyaan sub-tugas, seperti perilaku model sungguhan.
    """
    sql_replies = {
        question: list(reply) if isinstance(reply, list | tuple) else [reply]
        for question, reply in (sql or {}).items()
    }
    answers = answers or {}
    tools = {SQL_ANSWER_SYSTEM: "query_jobs", RAG_ANSWER_SYSTEM: "retrieve_jobs"}

    def reply(messages: list[BaseMessage]) -> Any:
        system = messages[0].content
        if system.startswith(SUPERVISOR_PLANNER_SYSTEM.split("{", 1)[0]):
            return plan
        if system.startswith(SQL_GENERATION_SYSTEM.split("{", 1)[0]):
            question = messages[-1].content.removeprefix("Pertanyaan: ").split("\n", 1)[0]
            return sql_replies[question].pop(0)
        if system in tools:
            question = messages[1].content
            if isinstance(messages[-1], ToolMessage):
                return answers[question]
            return tool_call(tools[system], question=question)
        return chat

    return reply
