"""Supervisor: satu-satunya pintu masuk UI ke agent, diorkestrasi dengan LangGraph.

Graph::

    START -> planner -> (Send per sub-tugas, paralel) -> sql_agent / rag_agent -> synthesize -> END
                     -> respond -> END            (tanpa sub-tugas: tanya balik, tolak, dll.)

- ``planner``: SATU panggilan LLM (structured output) yang memecah pertanyaan menjadi
  1-3 sub-tugas ``{agent, question}`` dan sekaligus menulis ulang pertanyaan lanjutan
  menjadi pertanyaan mandiri. Untuk pertanyaan yang tidak butuh data, planner memilih
  jenis respons (tanya balik, tolak, di luar topik, data tidak tersedia).
- ``sql_agent`` / ``rag_agent``: dijalankan paralel lewat ``Send``, satu sub-tugas per
  panggilan. ``run_sql_agent`` / ``run_rag_agent`` dipakai apa adanya, sehingga guard
  yang sudah ada (SQL read-only, sitasi [ID:x]) tetap berlaku per sub-tugas.
- ``synthesize``: TANPA LLM. Satu sub-tugas diteruskan apa adanya; lebih dari satu
  digabung dengan template (judul per bagian) sehingga angka dan ID tidak ditulis ulang.
- ``respond``: pesan tetap tanpa LLM untuk respons selain ``answer``, kecuali
  ``out_of_scope`` yang dibalas LLM secara luwes dengan guard (1 panggilan tambahan).

Kenapa orkestrasi ada di supervisor, bukan di agent: setiap agent tetap mengerjakan
satu tugas (mudah dites, guard tidak berubah). Baseline eval menunjukkan dua tugas
yang dipaksa masuk ke satu agent gagal (SQL dua statement, RAG satu pencarian untuk
dua posisi).

Kenapa planner + ``Send``, bukan supervisor LLM yang memanggil agent sebagai tool
(``langgraph_supervisor.create_supervisor``): pola tool-calling butuh satu panggilan
LLM per iterasi ditambah satu untuk jawaban akhir, sub-tugas berjalan berurutan, dan
jawaban agent ditulis ulang LLM (angka atau ID bisa berubah). Planner cukup satu
panggilan, sub-tugas paralel, dan biaya maksimal pasti (3 sub-tugas).

Dependency (``llm``, ``vector_store``, ``db_path``) disuntikkan lewat
``context_schema`` LangGraph, bukan di-import global. Graph cukup di-compile
sekali saat modul di-import, dan test bisa memakai LLM palsu tanpa API key.
"""

import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import BaseModel, Field

from agents.prompts import (
    SUPERVISOR_CHAT_SYSTEM,
    SUPERVISOR_CHAT_USER,
    SUPERVISOR_CLARIFY_MESSAGE,
    SUPERVISOR_DECLINED_NOTE,
    SUPERVISOR_FAILED_MESSAGE,
    SUPERVISOR_GENERAL_MESSAGE,
    SUPERVISOR_NO_HISTORY,
    SUPERVISOR_PLANNER_SYSTEM,
    SUPERVISOR_PLANNER_USER,
    SUPERVISOR_REFUSE_MESSAGE,
    SUPERVISOR_SKIPPED_NOTE,
    SUPERVISOR_UNSUPPORTED_MESSAGE,
)
from agents.rag_agent import run_rag_agent
from agents.sql_agent import run_sql_agent
from models.agent import AgentResult, LLMUsage

Agent = Literal["sql", "rag"]
Response = Literal["answer", "clarify", "refuse", "out_of_scope", "unsupported"]

MAX_SUBTASKS = 3
# Riwayat dipakai planner untuk menulis ulang pertanyaan lanjutan dan balasan luwes
# untuk pertanyaan tentang percakapan. 5 putaran terakhir (5 pengguna + 5 asisten).
MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_CHARS = 600

AGENT_TO_NODE: dict[str, str] = {"sql": "sql_agent", "rag": "rag_agent"}
FIXED_MESSAGES: dict[str, str] = {
    "clarify": SUPERVISOR_CLARIFY_MESSAGE,
    "refuse": SUPERVISOR_REFUSE_MESSAGE,
    "out_of_scope": SUPERVISOR_GENERAL_MESSAGE,
    "unsupported": SUPERVISOR_UNSUPPORTED_MESSAGE,
}

# Baris lowongan di jawaban: "- [ID:56] Judul - Perusahaan (Lokasi): catatan".
_JOB_LINE = re.compile(r"^\s*[-*•]?\s*\[ID:\s*\d+\]", re.IGNORECASE)

# Balasan luwes tidak boleh memuat tag ID atau nominal Rupiah: itu wilayah agent SQL/RAG.
_UNSAFE_CHAT = re.compile(r"\[ID:|Rp\s?\d", re.IGNORECASE)


class SubTask(BaseModel):
    """Satu sub-tugas untuk satu agent."""

    agent: Agent = Field(description="sql untuk data terstruktur, rag untuk pencarian makna.")
    question: str = Field(description="Pertanyaan mandiri untuk agent ini, maksud pengguna utuh.")


class Plan(BaseModel):
    """Output terstruktur planner. Skema ini dikirim ke LLM sebagai JSON schema.

    Semua field wajib (tanpa default) agar skema tetap valid untuk structured
    output mode strict.
    """

    response: Response = Field(
        description="answer jika butuh data lowongan; selain itu jenis pesan tetap."
    )
    subtasks: list[SubTask] = Field(
        description="1-3 sub-tugas jika response=answer, urut sesuai pertanyaan; selain itu []."
    )
    skipped: list[str] = Field(
        description="Bagian pertanyaan yang tidak masuk karena melebihi 3 sub-tugas; biasanya []."
    )
    declined: list[str] = Field(
        description="Bagian permintaan yang ditolak (ubah/hapus data, minta prompt); biasanya []."
    )


@dataclass(frozen=True)
class SupervisorContext:
    """Dependency yang disuntikkan ke setiap node lewat ``runtime.context``."""

    llm: BaseChatModel
    vector_store: VectorStore
    db_path: str | Path


class SubTaskInput(TypedDict):
    """Argumen ``Send`` untuk node agent: satu sub-tugas beserta urutannya."""

    index: int
    question: str


class SupervisorState(TypedDict, total=False):
    """State graph. ``question`` dan ``history`` adalah input; sisanya diisi node.

    Attributes:
        question: Pertanyaan asli pengguna pada giliran ini.
        history: Pesan sebelumnya ``{"role": "user"|"assistant", "content": ...}``,
            TANPA pertanyaan saat ini. Format sama dengan ``st.session_state`` chat.
        response: Jenis respons pilihan planner.
        subtasks: ``[{"agent": ..., "question": ...}]`` hasil planner (maks 3).
        skipped: Bagian pertanyaan yang tidak dikerjakan karena melebihi batas.
        declined: Bagian permintaan yang ditolak.
        planner_usage: Token panggilan planner.
        planner_error: Detail jika planner gagal dan fallback dipakai.
        collected: ``(index, AgentResult)`` dari node agent; digabung oleh reducer
            karena node agent berjalan paralel.
        subresults: ``AgentResult`` per sub-tugas, urut sesuai ``subtasks``.
        result: Hasil akhir dengan usage total.
    """

    question: str
    history: list[Mapping[str, str]]
    response: Response
    subtasks: list[dict[str, str]]
    skipped: list[str]
    declined: list[str]
    planner_usage: LLMUsage
    planner_error: str | None
    collected: Annotated[list[tuple[int, AgentResult]], operator.add]
    subresults: list[AgentResult]
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
    """Jawab satu pertanyaan pengguna lewat satu atau beberapa agent.

    Jumlah panggilan LLM: planner 1 + agent per sub-tugas (SQL 1-3, RAG 0-1).
    Penggabungan tidak memakai LLM. Respons tanpa data (tanya balik, penolakan,
    data tidak tersedia) hanya 1 panggilan (planner); ``out_of_scope`` (sapaan,
    pertanyaan riwayat, topik lain) 2 panggilan (planner + balasan luwes);
    pertanyaan kosong 0.

    Args:
        question: Pertanyaan pengguna pada giliran ini, boleh berupa pertanyaan
            lanjutan ("yang di Surabaya saja") atau gabungan beberapa permintaan.
        llm: Chat model untuk planner dan agent.
        vector_store: Vector store untuk RAG agent.
        db_path: Path ke file SQLite.
        history: Pesan-pesan sebelumnya, tidak termasuk ``question``.

    Returns:
        ``AgentResult``. Satu sub-tugas: hasil agent itu (``source`` sql/rag).
        Lebih dari satu, atau ada catatan: ``source="multi"`` dengan ``parts``
        per bagian. ``usage`` = planner + semua agent. Tidak pernah melempar exception.
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


def _unique_subtasks(tasks: Sequence[SubTask]) -> list[dict[str, str]]:
    """Sub-tugas planner tanpa pertanyaan kosong dan tanpa duplikat (agent + pertanyaan sama).

    Kenapa di Python, bukan aturan prompt: planner pernah mengeluarkan dua sub-tugas
    identik (uji manual 20 Okt 2026, "di mana saja loker selain di jakarta?"), sehingga
    jawaban tampil dua kali dan biaya LLM agent berlipat. Pembandingan mengabaikan huruf
    besar dan spasi berlebih; urutan kemunculan pertama dipertahankan.
    """
    seen: set[tuple[str, str]] = set()
    unique = []
    for task in tasks:
        question = task.question.strip()
        key = (task.agent, " ".join(question.lower().split()))
        if question and key not in seen:
            seen.add(key)
            unique.append({"agent": task.agent, "question": question})
    return unique


def plan_question(state: SupervisorState, runtime: Runtime[SupervisorContext]) -> dict:
    """Node planner: pecah pertanyaan menjadi sub-tugas dalam satu panggilan LLM.

    Kenapa structured output: nilai agent dan jenis respons dibatasi ``Literal``
    oleh JSON schema, dan pertanyaan mandiri per sub-tugas didapat tanpa panggilan
    kedua. Batas 3 sub-tugas ditegakkan di Python, bukan hanya di prompt; kelebihannya
    masuk ``skipped`` agar pengguna tahu bagian mana yang belum dijawab.
    """
    question = state["question"].strip()
    usage = LLMUsage()
    if not question:
        return _plan_update("out_of_scope", [], usage)

    messages = [
        SystemMessage(SUPERVISOR_PLANNER_SYSTEM.format(max_subtasks=MAX_SUBTASKS)),
        HumanMessage(
            SUPERVISOR_PLANNER_USER.format(
                history=format_history(state.get("history", [])), question=question
            )
        ),
    ]
    planner = runtime.context.llm.with_structured_output(Plan, include_raw=True)
    try:
        output = planner.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - tipe error provider LLM sangat beragam
        return _fallback(question, usage, f"Planner error: {exc}")

    usage.add(output["raw"], step="planner")
    plan: Plan | None = output["parsed"]
    if plan is None:
        return _fallback(question, usage, f"Planner output tidak valid: {output['parsing_error']}")
    if plan.response != "answer":
        return _plan_update(plan.response, [], usage)

    subtasks = _unique_subtasks(plan.subtasks)
    if not subtasks:
        return _fallback(question, usage, "Planner tidak memberi sub-tugas")
    skipped = [*plan.skipped, *(task["question"] for task in subtasks[MAX_SUBTASKS:])]
    return _plan_update(
        "answer", subtasks[:MAX_SUBTASKS], usage, skipped=skipped, declined=plan.declined
    )


def dispatch(state: SupervisorState) -> list[Send] | str:
    """Conditional edge: satu ``Send`` per sub-tugas (paralel), atau ke ``respond``."""
    if state["response"] != "answer":
        return "respond"
    return [
        Send(AGENT_TO_NODE[task["agent"]], SubTaskInput(index=index, question=task["question"]))
        for index, task in enumerate(state["subtasks"])
    ]


def run_sql_node(task: SubTaskInput, runtime: Runtime[SupervisorContext]) -> dict:
    """Node SQL agent untuk satu sub-tugas."""
    ctx = runtime.context
    result = run_sql_agent(task["question"], llm=ctx.llm, db_path=ctx.db_path)
    return {"collected": [(task["index"], result)]}


def run_rag_node(task: SubTaskInput, runtime: Runtime[SupervisorContext]) -> dict:
    """Node RAG agent untuk satu sub-tugas."""
    ctx = runtime.context
    result = run_rag_agent(
        task["question"], llm=ctx.llm, vector_store=ctx.vector_store, db_path=ctx.db_path
    )
    return {"collected": [(task["index"], result)]}


def respond(state: SupervisorState, runtime: Runtime[SupervisorContext]) -> dict:
    """Node respons tanpa data: tanya balik, tolak, di luar topik, data tidak tersedia.

    Kenapa tanpa LLM untuk ``clarify``, ``refuse``, ``unsupported``: pesan tetap tidak bisa
    berhalusinasi tentang lowongan (baseline: "Ada lowongan?" dijawab 5 lowongan acak).

    ``out_of_scope`` (sapaan, pertanyaan kemampuan, pertanyaan riwayat, topik lain)
    dibalas LLM supaya luwes dan bisa memakai riwayat percakapan. Risiko halusinasi
    ditutup dua lapis: prompt melarang menyebut lowongan/angka/ID, dan ``_chat_reply``
    membuang balasan yang memuat tag ID atau nominal Rupiah (kembali ke pesan tetap).
    """
    response = state["response"]
    if response == "out_of_scope" and state["question"].strip():
        chat_usage = LLMUsage()
        answer = _chat_reply(state, runtime.context.llm, chat_usage)
        # _with_totals menimpa usage hasil, jadi token balasan digabung lewat planner_usage.
        state = {
            **state,
            "planner_usage": state.get("planner_usage", LLMUsage()).merged(chat_usage),
        }
    else:
        answer = FIXED_MESSAGES[response]
    result = AgentResult(answer=answer, source="general")
    return {"subresults": [], "result": _with_totals(result, state, [])}


def synthesize(state: SupervisorState) -> dict:
    """Node penggabung TANPA LLM.

    - Satu sub-tugas tanpa catatan: hasil agent diteruskan apa adanya (sama seperti
      supervisor lama, sehingga UI dan guard tidak berubah).
    - Selain itu: ``source="multi"``, ``parts`` = satu bagian per sub-tugas dengan
      judul pertanyaan mandirinya, ditambah bagian catatan (pertanyaan yang dilewati
      atau ditolak). Jawaban teks = gabungan semua bagian, untuk riwayat chat.

    Kenapa template, bukan LLM: angka dan ID [ID:x] dari agent tidak boleh berubah,
    dan template tidak menambah token maupun latensi. LLM baru dipertimbangkan jika
    eval menunjukkan template kurang (ukurannya: angka dan ID tetap utuh).
    """
    subresults = [result for _, result in sorted(state.get("collected", []), key=lambda x: x[0])]
    notes = _notes(state.get("skipped", []), state.get("declined", []))
    if len(subresults) == 1 and not notes:
        return {"subresults": subresults, "result": _with_totals(subresults[0], state, subresults)}

    with_titles = len(subresults) > 1
    parts = [
        result.model_copy(update={"answer": f"**{task['question']}**\n{result.answer}"})
        if with_titles
        else result
        for task, result in zip(state["subtasks"], subresults, strict=True)
    ]
    if notes:
        parts.append(AgentResult(answer=notes, source="general"))
    combined = AgentResult(
        answer="\n\n".join(part.answer for part in parts),
        source="multi",
        job_ids=list(dict.fromkeys(i for result in subresults for i in result.job_ids)),
        parts=parts,
    )
    return {"subresults": subresults, "result": _with_totals(combined, state, subresults)}


# --- Helper ------------------------------------------------------------------


def _plan_update(
    response: Response,
    subtasks: list[dict[str, str]],
    usage: LLMUsage,
    *,
    skipped: Sequence[str] = (),
    declined: Sequence[str] = (),
    error: str | None = None,
) -> dict:
    """Update state dari planner dalam satu bentuk yang sama untuk semua cabang."""
    return {
        "response": response,
        "subtasks": subtasks,
        "skipped": list(skipped),
        "declined": list(declined),
        "planner_usage": usage,
        "planner_error": error,
    }


def _fallback(question: str, usage: LLMUsage, error: str) -> dict:
    """Planner gagal: satu sub-tugas RAG dengan pertanyaan asli.

    RAG dipilih karena pertanyaan lowongan tetap terjawab dengan guard sitasi, dan
    pertanyaan di luar topik berhenti di threshold retrieval dengan pesan tetap.
    """
    return _plan_update("answer", [{"agent": "rag", "question": question}], usage, error=error)


def _notes(skipped: Sequence[str], declined: Sequence[str]) -> str:
    """Catatan untuk bagian yang tidak dikerjakan; kosong jika tidak ada."""
    lines = []
    if declined:
        lines.append(SUPERVISOR_DECLINED_NOTE)
    if skipped:
        lines.append(SUPERVISOR_SKIPPED_NOTE + "\n" + "\n".join(f"- {item}" for item in skipped))
    return "\n\n".join(lines)


def _with_totals(
    result: AgentResult, state: SupervisorState, subresults: Sequence[AgentResult]
) -> AgentResult:
    """Usage total (planner + semua agent, urut sub-tugas) dan semua error digabung."""
    usage = state.get("planner_usage", LLMUsage())
    for sub in subresults:
        usage = usage.merged(sub.usage)
    errors = [state.get("planner_error"), *(sub.error for sub in subresults)]
    return result.model_copy(
        update={"usage": usage, "error": "; ".join(e for e in errors if e) or None}
    )


def _chat_reply(state: SupervisorState, llm: BaseChatModel, usage: LLMUsage) -> str:
    """Balasan luwes untuk ``out_of_scope``; satu panggilan LLM dengan riwayat percakapan.

    Kembali ke ``SUPERVISOR_GENERAL_MESSAGE`` jika LLM gagal, balasan kosong, atau
    balasan memuat tag ``[ID:x]`` / nominal Rupiah (bukan wilayah balasan umum).
    """
    history = state.get("history", [])
    previous_questions = format_previous_questions(history)
    messages = [
        SystemMessage(SUPERVISOR_CHAT_SYSTEM),
        HumanMessage(
            SUPERVISOR_CHAT_USER.format(
                previous_questions=previous_questions,
                history=format_history(history),
                question=state["question"].strip(),
            )
        ),
    ]
    try:
        response = llm.invoke(messages)
    except Exception:  # noqa: BLE001 - tipe error provider LLM sangat beragam
        return SUPERVISOR_GENERAL_MESSAGE
    usage.add(response, step="chat")
    text = response.text.strip()
    if not text:
        return SUPERVISOR_GENERAL_MESSAGE
    # Pola terlarang yang memang diketik pengguna (misalnya "gaji di atas Rp 8 juta")
    # boleh muncul saat pertanyaan itu ditulis ulang; selain itu dianggap karangan LLM.
    unsafe = _UNSAFE_CHAT.search(text)
    if unsafe and unsafe.group(0).lower() not in previous_questions.lower():
        return SUPERVISOR_GENERAL_MESSAGE
    return text


def _compact_assistant(content: str) -> str:
    """Ringkas jawaban asisten untuk riwayat: baris pembuka + identitas setiap lowongan.

    Planner butuh ID dan nama lowongan yang sudah ditampilkan ("yang dari Pandurasa",
    "yang kedua") untuk pertanyaan lanjutan. Alasan per lowongan dibuang agar ID
    lowongan ke-4 dan ke-5 tidak terpotong oleh batas karakter.
    """
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return ""
    kept = [lines[0]]
    for line in lines[1:]:
        if _JOB_LINE.match(line):
            head, separator, _ = line.partition("): ")
            kept.append(head + ")" if separator else line)
    return " ".join(kept)


def format_history(history: Sequence[Mapping[str, str]]) -> str:
    """Ubah riwayat chat menjadi teks pendek untuk prompt planner.

    Hanya ``MAX_HISTORY_MESSAGES`` pesan terakhir, masing-masing maksimal
    ``MAX_HISTORY_CHARS`` karakter. Jawaban asisten diringkas dulu dengan
    ``_compact_assistant`` sehingga ID lowongan tetap terbawa.
    """
    lines = []
    for message in list(history)[-MAX_HISTORY_MESSAGES:]:
        is_user = message.get("role") == "user"
        raw = str(message.get("content", ""))
        content = " ".join((raw if is_user else _compact_assistant(raw)).split())
        if len(content) > MAX_HISTORY_CHARS:
            content = f"{content[:MAX_HISTORY_CHARS]}..."
        lines.append(f"{'Pengguna' if is_user else 'Asisten'}: {content}")
    return "\n".join(lines) or SUPERVISOR_NO_HISTORY


def format_previous_questions(history: Sequence[Mapping[str, str]]) -> str:
    """Daftar bernomor pertanyaan pengguna dari riwayat, paling lama di atas.

    Kenapa dipisah dari ``format_history``: LLM balasan luwes pernah menjawab "belum ada
    pertanyaan sebelumnya" padahal riwayat lengkap ada di prompt. Dengan daftar yang
    disiapkan kode, "pertanyaan terakhir" = nomor terbesar, tidak perlu ditebak LLM.
    Batasnya sama dengan ``format_history``.
    """
    questions = []
    for message in list(history)[-MAX_HISTORY_MESSAGES:]:
        if message.get("role") != "user":
            continue
        content = " ".join(str(message.get("content", "")).split())
        if content:
            questions.append(
                f"{content[:MAX_HISTORY_CHARS]}..." if len(content) > MAX_HISTORY_CHARS else content
            )
    return "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1)) or SUPERVISOR_NO_HISTORY


# --- Graph -------------------------------------------------------------------


def build_supervisor_graph() -> CompiledStateGraph:
    """Rakit dan compile graph supervisor.

    Returns:
        Graph siap ``invoke(state, context=SupervisorContext(...))``.
    """
    graph = StateGraph(SupervisorState, context_schema=SupervisorContext)
    graph.add_node("planner", plan_question)
    graph.add_node("sql_agent", run_sql_node)
    graph.add_node("rag_agent", run_rag_node)
    graph.add_node("synthesize", synthesize)
    graph.add_node("respond", respond)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", dispatch, ["sql_agent", "rag_agent", "respond"])
    graph.add_edge("sql_agent", "synthesize")
    graph.add_edge("rag_agent", "synthesize")
    graph.add_edge("synthesize", END)
    graph.add_edge("respond", END)
    return graph.compile(name="career_assistant_supervisor")


SUPERVISOR_GRAPH = build_supervisor_graph()
