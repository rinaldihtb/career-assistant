"""Supervisor sebagai LangChain agent yang mendelegasikan worker sebagai tools."""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools.base import BaseTool
from langgraph.graph.state import CompiledStateGraph

from src.repositories.job_repository import JobRepository

SUPERVISOR_AGENT_SYSTEM = """You are a career-assistant supervisor.
Use ask_sql_agent for structured questions about counts, salaries, locations, and work types.
Use ask_rag_agent for semantic job search, skills, and candidate-job matching.
You may call one or both workers when the user asks multiple questions.
Do not invent job data; use a worker before answering data questions."""


class SupervisorAgent:
    """Supervisor fleksibel: LLM memutuskan worker/tool yang akan dipanggil."""

    def __init__(
        self,
        llm: BaseChatModel,
        job_repository: JobRepository,
        agent_tools: list[BaseTool],
    ) -> None:
        self.llm = llm
        self.job_repository = job_repository
        self.agent_tools = agent_tools
        self.run = self.create_supervisor_agent()

    def create_supervisor_agent(self) -> CompiledStateGraph:
        """Buat supervisor dengan RAG dan SQL sebagai LangChain agent tools."""

        return create_agent(
            model=self.llm,
            system_prompt=SUPERVISOR_AGENT_SYSTEM,
            tools=self.agent_tools,
            name="agent_supervisor",
        )
