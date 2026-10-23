"""Supervisor StateGraph yang memecah pertanyaan ke worker RAG dan SQL."""

import operator
import re
from typing import Annotated, Literal, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel

from agents.prompts import (
    SUPERVISOR_CLARIFY_MESSAGE,
    SUPERVISOR_DECLINED_NOTE,
    SUPERVISOR_GENERAL_MESSAGE,
    SUPERVISOR_PLANNER_SYSTEM,
    SUPERVISOR_REFUSE_MESSAGE,
    SUPERVISOR_SKIPPED_NOTE,
    SUPERVISOR_UNSUPPORTED_MESSAGE,
)
from agents.rag_agent import RetrievalResult
from agents.sql_agent import SQLQueryResult
from models.agent import AgentResult, LLMUsage
from repositories.job_repository import JobRepository

MAX_SUBTASKS = 3
AgentName = Literal["sql", "rag"]
Response = Literal["answer", "clarify", "refuse", "out_of_scope", "unsupported"]
_CITATION = re.compile(r"\[ID:\s*(\d+)\]", re.IGNORECASE)


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
    planner_usage: LLMUsage


class SupervisorState(TypedDict, total=False):
    """State bersama dari planner sampai jawaban akhir."""

    messages: Annotated[list[BaseMessage], add_messages]
    response: Response
    subtasks: list[dict[str, str]]
    skipped: list[str]
    declined: list[str]
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
        """Ubah satu pertanyaan menjadi satu atau beberapa sub-tugas worker."""
        messages = [
            SystemMessage(SUPERVISOR_PLANNER_SYSTEM.format(max_subtasks=MAX_SUBTASKS)),
            *state.get("messages", []),
        ]
        try:
            output = cast(
                dict[str, object],
                self.llm.with_structured_output(Plan, include_raw=True).invoke(messages),
            )
        except Exception:  # noqa: BLE001 - provider/parsing error beragam.
            return {
                "response": "clarify",
                "subtasks": [],
                "skipped": [],
                "declined": [],
                "planner_usage": LLMUsage(),
            }

        planner_usage = LLMUsage()
        raw = output.get("raw")
        if isinstance(raw, AIMessage):
            planner_usage.add(raw, step="planner")

        plan = output.get("parsed")
        if not isinstance(plan, Plan):
            return {
                "response": "clarify",
                "subtasks": [],
                "skipped": [],
                "declined": [],
                "planner_usage": planner_usage,
            }

        subtasks = [
            {"agent": task.agent, "question": task.question.strip()}
            for task in plan.subtasks
            if task.question.strip()
        ]
        response: Response = plan.response if subtasks or plan.response != "answer" else "clarify"
        return {
            "response": response,
            "subtasks": subtasks[:MAX_SUBTASKS],
            "skipped": plan.skipped,
            "declined": plan.declined,
            "planner_usage": planner_usage,
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
                answer=str(answer),
                source="sql",
                job_ids=sql_data.job_ids,
                sql=sql_data.sql,
                rows=sql_data.rows,
                usage=usage,
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
        """Kirim respons tetap jika planner tidak membutuhkan worker."""
        response: Response = state.get("response", "out_of_scope")
        return {
            "result": AgentResult(
                answer=FIXED_RESPONSES[response],
                source="general",
                usage=state.get("planner_usage", LLMUsage()),
            )
        }

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
