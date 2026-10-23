"""Dependency instances used by the application."""

from agents.rag_agent import RagAgent
from agents.sql_agent import SqlAgent

# from agents.supervisor_agent_normal import SupervisorAgent
from agents.supervisor_agent_graph import SupervisorAgent
from config.database import SQL_DB, VECTOR_DB
from config.llm import LLM_MODEL
from repositories.job_repository import JobRepository

job_repository = JobRepository(db=SQL_DB, vector_db=VECTOR_DB)

# Normal
# agent_tools = [
#     RagAgent(llm=LLM_MODEL, job_repository=job_repository).as_tool(),
#     SqlAgent(llm=LLM_MODEL, job_repository=job_repository).as_tool(),
# ]

# supervisor_agent = SupervisorAgent(
#     llm=LLM_MODEL,
#     job_repository=job_repository,
#     agent_tools=agent_tools,
# )

# Graph
supervisor_agent = SupervisorAgent(
    llm=LLM_MODEL,
    job_repository=job_repository,
    rag_agent=RagAgent(LLM_MODEL, job_repository).create_rag_agent(),
    sql_agent=SqlAgent(LLM_MODEL, job_repository).create_sql_agent(),
)
