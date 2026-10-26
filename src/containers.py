"""Composition root: membuat dependency aplikasi sekali per proses.

Konfigurasi LLM, Qdrant, dan SQLite di ``src/config`` baru di-import saat fungsi
dipanggil pertama kali, sehingga modul ini bisa di-import tanpa .env dan tanpa
jaringan. Test mengganti fungsi di modul pemakainya dengan dependency palsu.
"""

from functools import cache

from src.agents.rag_agent import RagAgent
from src.agents.sql_agent import SqlAgent
from src.agents.supervisor_agent_graph import SupervisorAgent
from src.repositories.job_repository import JobRepository


@cache
def get_job_repository() -> JobRepository:
    """Repository lowongan dari ``SQL_DB`` dan ``VECTOR_DB``; dibuat sekali lalu dipakai ulang."""
    from src.config.database import SQL_DB, VECTOR_DB

    return JobRepository(db=SQL_DB, vector_db=VECTOR_DB)


@cache
def get_supervisor_agent() -> SupervisorAgent:
    """Supervisor graph dengan worker RAG dan SQL yang memakai ``LLM_MODEL`` yang sama."""
    from src.config.llm import LLM_MODEL

    job_repository = get_job_repository()
    return SupervisorAgent(
        llm=LLM_MODEL,
        job_repository=job_repository,
        rag_agent=RagAgent(LLM_MODEL, job_repository).create_rag_agent(),
        sql_agent=SqlAgent(LLM_MODEL, job_repository).create_sql_agent(),
    )
