"""Supervisor: satu-satunya pintu masuk UI ke agent, diorkestrasi dengan LangGraph.

Graph::

    START -> router -> sql_agent -> finalize -> END
                    -> rag_agent ->
                    -> general   ->

- ``router``: SATU panggilan LLM (structured output) yang memilih rute DAN
  menulis ulang pertanyaan lanjutan menjadi pertanyaan mandiri.
- ``sql_agent`` / ``rag_agent``: memanggil ``run_sql_agent`` / ``run_rag_agent``
  apa adanya. Jawaban agent TIDAK dirangkum ulang oleh LLM supervisor, supaya
  guard yang sudah ada (SQL read-only, sitasi [ID:x]) tidak dirusak.
- ``general``: sapaan atau di luar topik dijawab pesan tetap, 0 panggilan LLM.
- ``finalize``: menjumlahkan token router + agent dan menggabungkan error.

Kenapa tidak memakai ``langgraph_supervisor.create_supervisor``: pola tersebut
membuat LLM supervisor memanggil agent sebagai tool lalu menulis jawaban akhir
sendiri. Akibatnya 2 panggilan LLM tambahan per pertanyaan dan jawaban agent
bisa berubah (angka atau ID lowongan ikut ditulis ulang).

Dependency (``llm``, ``vector_store``, ``db_path``) disuntikkan lewat
``context_schema`` LangGraph, bukan di-import global. Graph cukup di-compile
sekali saat modul di-import, dan test bisa memakai LLM palsu tanpa API key.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from agents.prompts import (
    SUPERVISOR_FAILED_MESSAGE,
    SUPERVISOR_GENERAL_MESSAGE,
    SUPERVISOR_NO_HISTORY,
    SUPERVISOR_ROUTER_SYSTEM,
    SUPERVISOR_ROUTER_USER,
)
from agents.rag_agent import run_rag_agent
from agents.sql_agent import run_sql_agent
from models.agent import AgentResult, LLMUsage

Route = Literal["sql", "rag", "general"]

# Jika router gagal (error API atau output tidak valid), pertanyaan diteruskan
# ke RAG: pertanyaan lowongan tetap terjawab dengan guard sitasi, dan pertanyaan
# di luar topik berhenti di threshold retrieval dengan pesan tetap (0 LLM call).
FALLBACK_ROUTE: Route = "rag"
# Riwayat hanya dipakai untuk menulis ulang pertanyaan lanjutan, jadi 3 putaran
# terakhir dan potongan pendek per pesan sudah cukup; token router tetap kecil.
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 300

ROUTE_TO_NODE: dict[Route, str] = {"sql": "sql_agent", "rag": "rag_agent", "general": "general"}


class RouteDecision(BaseModel):
    """Output terstruktur router. Skema ini dikirim ke LLM sebagai JSON schema."""

    route: Route = Field(description="Agent yang menjawab: sql, rag, atau general.")
    question: str = Field(
        description="Pertanyaan terakhir pengguna yang ditulis ulang menjadi kalimat mandiri."
    )


@dataclass(frozen=True)
class SupervisorContext:
    """Dependency yang disuntikkan ke setiap node lewat ``runtime.context``."""

    llm: BaseChatModel
    vector_store: VectorStore
    db_path: str | Path


class SupervisorState(TypedDict, total=False):
    """State graph. ``question`` dan ``history`` adalah input; sisanya diisi node.

    Attributes:
        question: Pertanyaan asli pengguna pada giliran ini.
        history: Pesan sebelumnya ``{"role": "user"|"assistant", "content": ...}``,
            TANPA pertanyaan saat ini. Format sama dengan ``st.session_state`` chat.
        route: Rute pilihan router.
        standalone_question: Pertanyaan mandiri yang diteruskan ke agent.
        router_usage: Token panggilan router.
        router_error: Detail jika router gagal dan fallback dipakai.
        result: Hasil akhir; setelah ``finalize`` berisi usage total.
    """

    question: str
    history: list[Mapping[str, str]]
    route: Route
    standalone_question: str
    router_usage: LLMUsage
    router_error: str | None
    result: AgentResult


# --- Entry point -------------------------------------------------------------


def run_supervisor(
    question: str,
    *,
    llm: BaseChatModel,
    vector_store: VectorStore,
    db_path: str | Path,
    history: Sequence[Mapping[str, str]] = (),
) -> AgentResult:
    """Jawab satu pertanyaan pengguna lewat agent yang tepat.

    Jumlah panggilan LLM: router 1 + agent terpilih (SQL 1-3, RAG 0-1, general 0).

    Args:
        question: Pertanyaan pengguna pada giliran ini, boleh berupa pertanyaan
            lanjutan ("yang di Surabaya saja").
        llm: Chat model untuk router dan agent.
        vector_store: Vector store untuk RAG agent.
        db_path: Path ke file SQLite.
        history: Pesan-pesan sebelumnya, tidak termasuk ``question``.

    Returns:
        ``AgentResult`` dari agent terpilih, dengan ``usage`` = router + agent.
        Fungsi ini tidak pernah melempar exception.
    """
    context = SupervisorContext(llm=llm, vector_store=vector_store, db_path=db_path)
    try:
        state = SUPERVISOR_GRAPH.invoke(
            {"question": question, "history": list(history)}, context=context
        )
        return state["result"]
    except Exception as exc:  # noqa: BLE001 - kontrak: supervisor tidak pernah melempar exception
        return AgentResult(
            answer=SUPERVISOR_FAILED_MESSAGE, source="general", error=f"Supervisor error: {exc}"
        )


# --- Node --------------------------------------------------------------------


def route_question(state: SupervisorState, runtime: Runtime[SupervisorContext]) -> dict:
    """Node router: pilih rute dan tulis ulang pertanyaan dalam satu panggilan LLM.

    Kenapa structured output, bukan teks bebas "SQL"/"RAG": nilai rute dibatasi
    ``Literal`` oleh JSON schema, dan pertanyaan mandiri ikut didapat tanpa
    panggilan kedua. Jika output tetap tidak valid, ``FALLBACK_ROUTE`` dipakai.
    """
    question = state["question"].strip()
    usage = LLMUsage()
    if not question:
        return {"route": "general", "standalone_question": "", "router_usage": usage}

    messages = [
        SystemMessage(SUPERVISOR_ROUTER_SYSTEM),
        HumanMessage(
            SUPERVISOR_ROUTER_USER.format(
                history=format_history(state.get("history", [])), question=question
            )
        ),
    ]
    router = runtime.context.llm.with_structured_output(RouteDecision, include_raw=True)
    try:
        output = router.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - tipe error provider LLM sangat beragam
        return _fallback(question, usage, f"Router error: {exc}")

    usage.add(output["raw"], step="router")
    decision: RouteDecision | None = output["parsed"]
    if decision is None:
        return _fallback(question, usage, f"Router output tidak valid: {output['parsing_error']}")
    return {
        "route": decision.route,
        "standalone_question": decision.question.strip() or question,
        "router_usage": usage,
    }


def run_sql_node(state: SupervisorState, runtime: Runtime[SupervisorContext]) -> dict:
    """Node SQL agent: pertanyaan statistik."""
    ctx = runtime.context
    result = run_sql_agent(state["standalone_question"], llm=ctx.llm, db_path=ctx.db_path)
    return {"result": result}


def run_rag_node(state: SupervisorState, runtime: Runtime[SupervisorContext]) -> dict:
    """Node RAG agent: pencarian lowongan."""
    ctx = runtime.context
    result = run_rag_agent(
        state["standalone_question"],
        llm=ctx.llm,
        vector_store=ctx.vector_store,
        db_path=ctx.db_path,
    )
    return {"result": result}


def answer_general(state: SupervisorState) -> dict:
    """Node general: pesan tetap tanpa LLM untuk sapaan dan pertanyaan di luar topik.

    Kenapa tanpa LLM: jawaban ini tidak butuh data, dan pesan tetap tidak bisa
    berhalusinasi tentang lowongan.
    """
    return {"result": AgentResult(answer=SUPERVISOR_GENERAL_MESSAGE, source="general")}


def finalize(state: SupervisorState) -> dict:
    """Node akhir: usage total router + agent, error keduanya digabung."""
    result = state["result"]
    errors = [error for error in (state.get("router_error"), result.error) if error]
    return {
        "result": result.model_copy(
            update={
                "usage": state.get("router_usage", LLMUsage()).merged(result.usage),
                "error": "; ".join(errors) or None,
            }
        )
    }


def select_agent(state: SupervisorState) -> str:
    """Conditional edge: nama node tujuan berdasarkan ``state["route"]``."""
    return ROUTE_TO_NODE[state["route"]]


# --- Helper ------------------------------------------------------------------


def _fallback(question: str, usage: LLMUsage, error: str) -> dict:
    """Update state saat router gagal: rute cadangan dengan pertanyaan asli."""
    return {
        "route": FALLBACK_ROUTE,
        "standalone_question": question,
        "router_usage": usage,
        "router_error": error,
    }


def format_history(history: Sequence[Mapping[str, str]]) -> str:
    """Ubah riwayat chat menjadi teks pendek untuk prompt router.

    Hanya ``MAX_HISTORY_MESSAGES`` pesan terakhir, masing-masing dipotong
    ``MAX_HISTORY_CHARS`` karakter. Jawaban asisten bisa panjang (daftar 5
    lowongan), padahal router hanya butuh topiknya.
    """
    lines = []
    for message in list(history)[-MAX_HISTORY_MESSAGES:]:
        speaker = "Pengguna" if message.get("role") == "user" else "Asisten"
        content = " ".join(str(message.get("content", "")).split())
        if len(content) > MAX_HISTORY_CHARS:
            content = f"{content[:MAX_HISTORY_CHARS]}..."
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) or SUPERVISOR_NO_HISTORY


# --- Graph -------------------------------------------------------------------


def build_supervisor_graph() -> CompiledStateGraph:
    """Rakit dan compile graph supervisor.

    Returns:
        Graph siap ``invoke(state, context=SupervisorContext(...))``.
    """
    graph = StateGraph(SupervisorState, context_schema=SupervisorContext)
    graph.add_node("router", route_question)
    graph.add_node("sql_agent", run_sql_node)
    graph.add_node("rag_agent", run_rag_node)
    graph.add_node("general", answer_general)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", select_agent, list(ROUTE_TO_NODE.values()))
    for node in ROUTE_TO_NODE.values():
        graph.add_edge(node, "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(name="career_assistant_supervisor")


SUPERVISOR_GRAPH = build_supervisor_graph()
