"""RagAgent lengkap (create_agent) dengan LLM palsu dan vector store stub.

Yang dites di sini adalah alur di sekitar LLM: tool dipanggil dengan pertanyaan
sub-tugas, hasil retrieval tersimpan sebagai artifact, dan jawaban akhir diambil
dari panggilan LLM setelah tool. Guard sitasi ada di supervisor (test_supervisor.py).
"""

from conftest import ScriptedLLM, agent_script
from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.prompts import RAG_ANSWER_SYSTEM
from src.agents.rag_agent import RagAgent, RetrievalResult

CHUNKS = [
    (1, "Analisis data penjualan dengan SQL dan Python setiap minggu.", 0.72),
    (2, "Membantu cleaning data dan membuat dashboard untuk tim.", 0.65),
]


def test_agent_calls_retrieve_tool_then_answers(make_repository):
    question = "Lowongan data analyst"
    answer = "Ada dua:\n- [ID:1] Data Analyst - PT Alpha (Jakarta): cocok."
    llm = ScriptedLLM(reply=agent_script(answers={question: answer}))
    repository = make_repository(CHUNKS)

    state = (
        RagAgent(llm, repository).create_rag_agent().invoke({"messages": [HumanMessage(question)]})
    )

    assert state["messages"][-1].content == answer
    tool_message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert tool_message.name == "retrieve_jobs"
    assert isinstance(tool_message.artifact, RetrievalResult)
    assert [job["id"] for job in tool_message.artifact.jobs] == [1, 2]
    assert repository.vector_db.calls[0]["query"] == question
    assert len(llm.received) == 2
    assert all(messages[0].content == RAG_ANSWER_SYSTEM for messages in llm.received)


def test_final_llm_call_sees_retrieved_jobs_as_tool_result(make_repository):
    question = "Lowongan data analyst"
    llm = ScriptedLLM(reply=agent_script(answers={question: "ok"}))
    RagAgent(llm, make_repository(CHUNKS)).create_rag_agent().invoke(
        {"messages": [HumanMessage(question)]}
    )

    tool_result = llm.received[-1][-1]
    assert isinstance(tool_result, ToolMessage)
    assert '"job_title":"Data Analyst"' in tool_result.content
