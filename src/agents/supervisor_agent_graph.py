"""Supervisor StateGraph yang memecah pertanyaan ke worker RAG dan SQL."""

import operator
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel

from src.agents.prompts import (
    SUPERVISOR_CHAT_SYSTEM,
    SUPERVISOR_CHAT_USER,
    SUPERVISOR_CLARIFY_MESSAGE,
    SUPERVISOR_CLARIFY_SYSTEM,
    SUPERVISOR_DECLINED_NOTE,
    SUPERVISOR_GENERAL_MESSAGE,
    SUPERVISOR_NO_HISTORY,
    SUPERVISOR_PLANNER_SYSTEM,
    SUPERVISOR_PLANNER_USER,
    SUPERVISOR_REFUSE_MESSAGE,
    SUPERVISOR_SKIPPED_NOTE,
    SUPERVISOR_UNSUPPORTED_MESSAGE,
)
from src.agents.rag_agent import RetrievalResult
from src.agents.sql_agent import SQLQueryResult, finalize_answer
from src.models.agent import AgentResult, LLMUsage
from src.repositories.job_repository import JobRepository

MAX_SUBTASKS = 3
AgentName = Literal["sql", "rag"]
Response = Literal["answer", "clarify", "refuse", "out_of_scope", "unsupported"]
_CITATION = re.compile(r"\[ID:\s*(\d+)\]", re.IGNORECASE)
_UNSAFE_CHAT = re.compile(r"\[ID:|Rp\s?\d", re.IGNORECASE)
MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_CHARS = 600


class SubTask(BaseModel):
    """Satu pekerjaan yang diberikan planner kepada satu worker."""

    agent: AgentName
    question: str


class Plan(BaseModel):
    """Output structured dari planner."""

    response: Response
    subtasks: list[SubTask]
    skipped: list[str]
    declined: list[str]


class SupervisorState(TypedDict, total=False):
    """State bersama dari planner sampai jawaban akhir."""

    messages: Annotated[list[BaseMessage], add_messages]
    response: Response
    subtasks: list[dict[str, str]]
    skipped: list[str]
    declined: list[str]
    planner_usage: LLMUsage
    worker_index: int
    worker_question: str
    collected: Annotated[list[tuple[int, AgentResult]], operator.add]
    subresults: list[AgentResult]
    result: AgentResult


FIXED_RESPONSES = {
    "clarify": SUPERVISOR_CLARIFY_MESSAGE,
    "refuse": SUPERVISOR_REFUSE_MESSAGE,
    "out_of_scope": SUPERVISOR_GENERAL_MESSAGE,
    "unsupported": SUPERVISOR_UNSUPPORTED_MESSAGE,
}


def _notes(skipped: list[str], declined: list[str]) -> str:
    """Buat catatan untuk bagian pertanyaan yang tidak dikerjakan."""
    notes = []
    if declined:
        notes.append(SUPERVISOR_DECLINED_NOTE)
    if skipped:
        notes.append(SUPERVISOR_SKIPPED_NOTE + "\n" + "\n".join(f"- {item}" for item in skipped))
    return "\n\n".join(notes)


def check_citations(answer: str, candidate_ids: list[int]) -> tuple[list[int], list[int]]:
    """Pisahkan ID yang dikutip LLM menjadi kandidat valid dan ID di luar kandidat."""
    candidates = set(candidate_ids)
    cited = list(dict.fromkeys(int(match) for match in _CITATION.findall(answer)))
    return (
        [job_id for job_id in cited if job_id in candidates],
        [job_id for job_id in cited if job_id not in candidates],
    )


def usage_from_messages(messages: list[BaseMessage], step: str) -> LLMUsage:
    """Catat token dari setiap respons LLM dalam satu worker."""
    usage = LLMUsage()
    for message in messages:
        if isinstance(message, AIMessage):
            usage.add(message, step=step)
    return usage


def _chat_context(messages: Sequence[BaseMessage]) -> tuple[str, str, str]:
    """Pisahkan pertanyaan terakhir dari riwayat untuk prompt percakapan umum."""
    items = list(messages)[-MAX_HISTORY_MESSAGES:]
    last_user_index = next(
        (index for index in range(len(items) - 1, -1, -1) if _message_role(items[index]) == "user"),
        None,
    )
    if last_user_index is None:
        return "", SUPERVISOR_NO_HISTORY, SUPERVISOR_NO_HISTORY

    question = _message_content(items[last_user_index])
    history_items = items[:last_user_index]
    history_lines = [
        f"{'Pengguna' if _message_role(item) == 'user' else 'Asisten'}: {_message_content(item)}"
        for item in history_items
        if _message_content(item)
    ]
    questions = [
        _message_content(item)
        for item in history_items
        if _message_role(item) == "user" and _message_content(item)
    ]
    history = "\n".join(history_lines) or SUPERVISOR_NO_HISTORY
    previous_questions = (
        "\n".join(f"{index}. {item}" for index, item in enumerate(questions, 1))
        or SUPERVISOR_NO_HISTORY
    )
    return question, history, previous_questions


def _message_role(message: BaseMessage | Mapping[str, object]) -> Literal["user", "assistant"]:
    """Normalisasi peran pesan dari objek LangChain atau dict riwayat UI."""
    if isinstance(message, Mapping):
        return "user" if message.get("role") in {"user", "human"} else "assistant"
    return "user" if message.type == "human" else "assistant"


def _message_content(message: BaseMessage | Mapping[str, object]) -> str:
    """Ambil konten pesan, rapikan spasi, dan batasi panjang prompt."""
    content = message.get("content", "") if isinstance(message, Mapping) else message.content
    text = " ".join(str(content).split())
    return f"{text[:MAX_HISTORY_CHARS]}..." if len(text) > MAX_HISTORY_CHARS else text


class SupervisorAgent:
    """Planner memilih satu atau beberapa worker, lalu graph menggabungkan jawabannya."""

    def __init__(
        self,
        llm: BaseChatModel,
        job_repository: JobRepository,
        rag_agent: CompiledStateGraph,
        sql_agent: CompiledStateGraph,
    ) -> None:
        self.job_repository = job_repository
        self.rag_agent = rag_agent
        self.sql_agent = sql_agent
        self.llm = llm
        self.graph = self.create_supervisor_agent()

    def planner(self, state: SupervisorState) -> dict:
        """Ubah pesan pengguna terakhir menjadi satu atau beberapa sub-tugas worker.

        Pertanyaan terakhir dan riwayat berlabel Pengguna/Asisten diambil dari
        ``messages`` (``_chat_context``) dan dikirim dalam satu pesan
        ``SUPERVISOR_PLANNER_USER``. Sub-tugas di atas ``MAX_SUBTASKS`` masuk ``skipped``
        agar pengguna diberi tahu bagian yang belum dijawab.
        """
        question, history, _ = _chat_context(state.get("messages", []))
        messages = [
            SystemMessage(SUPERVISOR_PLANNER_SYSTEM.format(max_subtasks=MAX_SUBTASKS)),
            HumanMessage(SUPERVISOR_PLANNER_USER.format(history=history, question=question)),
        ]
        fallback: dict = {"response": "clarify", "subtasks": [], "skipped": [], "declined": []}
        try:
            output = cast(
                dict[str, object],
                self.llm.with_structured_output(Plan, include_raw=True).invoke(messages),
            )
        except Exception:  # noqa: BLE001 - provider/parsing error beragam.
            return {**fallback, "planner_usage": LLMUsage()}

        usage = LLMUsage()
        usage.add(output.get("raw"), step="planner")
        parsed_plan = output.get("parsed")
        if not isinstance(parsed_plan, Plan):
            return {**fallback, "planner_usage": usage}

        subtasks = [
            {"agent": task.agent, "question": task.question.strip()}
            for task in parsed_plan.subtasks
            if task.question.strip()
        ]
        response: Response = (
            parsed_plan.response if subtasks or parsed_plan.response != "answer" else "clarify"
        )
        overflow = [task["question"] for task in subtasks[MAX_SUBTASKS:]]
        return {
            "response": response,
            "subtasks": subtasks[:MAX_SUBTASKS],
            "skipped": [*parsed_plan.skipped, *overflow],
            "declined": parsed_plan.declined,
            "planner_usage": usage,
        }

    def dispatch(self, state: SupervisorState) -> list[Send] | str:
        """Kirim setiap sub-tugas ke worker yang sesuai, atau langsung merespons."""
        response: Response = state.get("response", "out_of_scope")
        if response != "answer":
            return "respond"

        return [
            Send(
                "sql_agent" if task["agent"] == "sql" else "rag_agent",
                {"worker_index": index, "worker_question": task["question"]},
            )
            for index, task in enumerate(state.get("subtasks", []))
        ]

    def run_rag_agent(self, state: SupervisorState) -> dict[str, list[tuple[int, AgentResult]]]:
        """Jalankan worker RAG untuk satu sub-tugas."""
        index = state.get("worker_index")
        question = state.get("worker_question")
        if index is None or not question:
            return {"collected": []}
        result = self.rag_agent.invoke({"messages": [HumanMessage(content=question)]})
        answer = result["messages"][-1].content
        usage = usage_from_messages(result["messages"], step="rag")
        retrieval = next(
            (
                message.artifact
                for message in reversed(result["messages"])
                if isinstance(message, ToolMessage)
                and message.name == "retrieve_jobs"
                and isinstance(message.artifact, RetrievalResult)
            ),
            None,
        )
        if retrieval is None:
            return {
                "collected": [(index, AgentResult(answer=str(answer), source="rag", usage=usage))]
            }

        valid_ids, invalid_ids = check_citations(str(answer), [job["id"] for job in retrieval.jobs])
        jobs_by_id = {job["id"]: job for job in retrieval.jobs}
        return {
            "collected": [
                (
                    index,
                    AgentResult(
                        answer=str(answer),
                        source="rag",
                        job_ids=valid_ids,
                        rows=[jobs_by_id[job_id] for job_id in valid_ids],
                        usage=usage,
                        error=(
                            f"ID di luar kandidat dibuang: {invalid_ids}" if invalid_ids else None
                        ),
                    ),
                )
            ]
        }

    def run_sql_agent(self, state: SupervisorState) -> dict[str, list[tuple[int, AgentResult]]]:
        """Jalankan worker SQL untuk satu sub-tugas."""
        index = state.get("worker_index")
        question = state.get("worker_question")
        if index is None or not question:
            return {"collected": []}
        result = self.sql_agent.invoke({"messages": [HumanMessage(content=question)]})
        answer = result["messages"][-1].content
        usage = usage_from_messages(result["messages"], step="sql")
        sql_data = next(
            (
                message.artifact
                for message in reversed(result["messages"])
                if isinstance(message, ToolMessage)
                and message.name == "query_jobs"
                and isinstance(message.artifact, SQLQueryResult)
            ),
            None,
        )
        if sql_data is not None:
            sql_result = AgentResult(
                answer=finalize_answer(str(answer), sql_data.rows),
                source="sql",
                job_ids=sql_data.job_ids,
                sql=sql_data.sql,
                rows=sql_data.rows,
                usage=usage.merged(sql_data.usage),
                error=sql_data.error,
            )
        else:
            sql_result = AgentResult(answer=str(answer), source="sql", usage=usage)
        return {"collected": [(index, sql_result)]}

    def synthesize(self, state: SupervisorState) -> dict[str, AgentResult | list[AgentResult]]:
        """Gabungkan jawaban worker sesuai urutan pertanyaan."""
        subresults = [result for _, result in sorted(state.get("collected", []))]
        notes = _notes(state.get("skipped", []), state.get("declined", []))
        total_usage = state.get("planner_usage", LLMUsage())
        for result in subresults:
            total_usage = total_usage.merged(result.usage)

        if len(subresults) == 1 and not notes:
            result = subresults[0].model_copy(update={"usage": total_usage})
            return {"subresults": subresults, "result": result}

        tasks = state.get("subtasks", [])
        parts = [
            result.model_copy(update={"answer": f"**{task['question']}**\n{result.answer}"})
            if len(subresults) > 1
            else result
            for task, result in zip(tasks, subresults, strict=True)
        ]
        if notes:
            parts.append(AgentResult(answer=notes, source="general"))

        job_ids = list(dict.fromkeys(job_id for result in subresults for job_id in result.job_ids))
        return {
            "subresults": subresults,
            "result": AgentResult(
                answer="\n\n".join(part.answer for part in parts),
                source="multi",
                job_ids=job_ids,
                usage=total_usage,
                parts=parts,
            ),
        }

    def respond(self, state: SupervisorState) -> dict[str, AgentResult]:
        """Kirim respons tetap, atau balasan LLM yang luwes untuk percakapan umum."""
        response: Response = state.get("response", "out_of_scope")
        usage = state.get("planner_usage", LLMUsage())
        if response == "out_of_scope":
            chat_usage = LLMUsage()
            answer = self._chat_reply(
                state,
                chat_usage,
                system_prompt=SUPERVISOR_CHAT_SYSTEM,
                fallback=SUPERVISOR_GENERAL_MESSAGE,
            )
            usage = usage.merged(chat_usage)
        elif response == "clarify":
            chat_usage = LLMUsage()
            answer = self._chat_reply(
                state,
                chat_usage,
                system_prompt=SUPERVISOR_CLARIFY_SYSTEM,
                fallback=SUPERVISOR_CLARIFY_MESSAGE,
            )
            usage = usage.merged(chat_usage)
        else:
            answer = FIXED_RESPONSES[response]
        return {
            "result": AgentResult(
                answer=answer,
                source="general",
                usage=usage,
            )
        }

    def _chat_reply(
        self,
        state: SupervisorState,
        usage: LLMUsage,
        *,
        system_prompt: str,
        fallback: str,
    ) -> str:
        """Buat balasan LLM dengan riwayat percakapan dan fallback yang aman."""
        question, history, previous_questions = _chat_context(state.get("messages", []))
        if not question:
            return fallback

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(system_prompt),
                    HumanMessage(
                        SUPERVISOR_CHAT_USER.format(
                            previous_questions=previous_questions,
                            history=history,
                            question=question,
                        )
                    ),
                ]
            )
        except Exception:  # noqa: BLE001 - error provider beragam.
            return fallback

        usage.add(response, step="chat")
        answer = str(getattr(response, "text", response.content)).strip()
        if not answer or _UNSAFE_CHAT.search(answer):
            return fallback
        return answer

    def create_supervisor_agent(self) -> CompiledStateGraph:
        """Bangun workflow supervisor.

        Alur utama::

            START → planner
                    ├→ respond → END
                    ├→ rag_agent ─┐
                    └→ sql_agent ─┴→ synthesize → END

        ``planner`` memilih apakah pertanyaan perlu dijawab dengan pesan tetap
        atau dipecah menjadi sub-tugas RAG dan SQL. ``dispatch`` dapat mengirim
        beberapa sub-tugas sekaligus melalui ``Send``. Hasil setiap worker
        dikumpulkan oleh ``synthesize`` sebelum graph selesai.
        """
        graph = StateGraph(SupervisorState)
        graph.add_node("planner", self.planner)
        graph.add_node("rag_agent", self.run_rag_agent)
        graph.add_node("sql_agent", self.run_sql_agent)
        graph.add_node("synthesize", self.synthesize)
        graph.add_node("respond", self.respond)
        graph.add_edge(START, "planner")
        graph.add_conditional_edges("planner", self.dispatch)
        graph.add_edge("rag_agent", "synthesize")
        graph.add_edge("sql_agent", "synthesize")
        graph.add_edge("synthesize", END)
        graph.add_edge("respond", END)
        return graph.compile()


if __name__ == "__main__":
    from src.containers import get_supervisor_agent

    state = get_supervisor_agent().graph.invoke(
        {"messages": [HumanMessage(content="Lowongan data analyst jakarta?")]}
    )
    print(state["result"].answer)
