"""Fixture bersama untuk semua test.

Test memakai database kecil buatan sendiri, bukan data/database.sqlite,
supaya hasilnya pasti (tidak berubah saat data di-seed ulang) dan cepat.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field

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
]


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
def fake_vector_store():
    """Fixture pabrik stub: ``fake_vector_store([(job_id, teks, skor), ...])``."""
    return FakeVectorStore
