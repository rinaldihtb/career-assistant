"""Composition root: membuat dependency aplikasi sekali per proses.

Konfigurasi LLM, Qdrant, dan SQLite di ``src/config`` baru di-import saat fungsi
dipanggil pertama kali, sehingga modul ini bisa di-import tanpa .env dan tanpa
jaringan. Test mengganti fungsi di modul pemakainya dengan dependency palsu.
"""

import os
from functools import cache

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

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


@cache
def get_trace_callbacks() -> list[BaseCallbackHandler]:
    """Callback Langfuse untuk ``config`` LangChain; kosong jika ``LANGFUSE_PUBLIC_KEY`` kosong.

    Input, output, dan metadata setiap observasi dikosongkan sebelum dikirim karena state graph
    dan prompt bisa memuat profil serta kontak CV. Langfuse hanya menerima nama langkah, model,
    token, dan latensi.
    """
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return []
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    # Client dibuat lebih dulu agar CallbackHandler memakai client yang sama beserta mask-nya.
    Langfuse(mask=lambda *, data, **_: None)
    return [CallbackHandler()]


def traced_llm(llm: BaseChatModel, step: str) -> BaseChatModel:
    """Salinan ``llm`` yang tercatat di Langfuse dengan nama ``step``; ``llm`` jika tidak aktif.

    LangChain menyimpan serialisasi model (termasuk nama) saat pemanggilan pertama dan salinan
    ikut membawanya, jadi ``llm`` sebaiknya model yang tidak dipanggil langsung.
    """
    callbacks = get_trace_callbacks()
    return llm.model_copy(update={"callbacks": callbacks, "name": step}) if callbacks else llm
