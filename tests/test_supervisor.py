"""SupervisorAgent (planner + Send paralel + synthesize/respond) dengan LLM palsu.

Sub-tugas berjalan paralel, jadi ``ScriptedLLM`` memilih balasan dari isi prompt
(jenis prompt dan pertanyaan sub-tugas), bukan dari urutan panggilan. Balasan
planner berupa tool call ``Plan``, bentuk yang dibaca parser structured output.
"""

import pytest
from conftest import ScriptedLLM, agent_script, plan_reply, tool_call

from src.agents.prompts import (
    SUPERVISOR_CHAT_SYSTEM,
    SUPERVISOR_CLARIFY_MESSAGE,
    SUPERVISOR_CLARIFY_SYSTEM,
    SUPERVISOR_DECLINED_NOTE,
    SUPERVISOR_GENERAL_MESSAGE,
    SUPERVISOR_NO_HISTORY,
    SUPERVISOR_PLANNER_USER,
    SUPERVISOR_REFUSE_MESSAGE,
    SUPERVISOR_SKIPPED_NOTE,
    SUPERVISOR_UNSUPPORTED_MESSAGE,
)
from src.agents.rag_agent import RagAgent
from src.agents.sql_agent import SqlAgent
from src.agents.supervisor_agent_graph import SupervisorAgent, _chat_context, check_citations

LONG = "Deskripsi pekerjaan yang cukup panjang untuk lolos batas chunk minimum."
CHUNKS = [(1, LONG, 0.70), (2, LONG, 0.60)]
COUNT_SQL = "SELECT COUNT(*) AS jumlah FROM jobs WHERE lower(location) LIKE '%jakarta%'"
SQL_Q = "Berapa lowongan di Jakarta?"
RAG_Q = "Lowongan data analyst"


def _supervisor(llm, repository) -> SupervisorAgent:
    return SupervisorAgent(
        llm=llm,
        job_repository=repository,
        rag_agent=RagAgent(llm, repository).create_rag_agent(),
        sql_agent=SqlAgent(llm, repository).create_sql_agent(),
    )


def _invoke(llm, repository, question, history=()):
    messages = [*history, {"role": "user", "content": question}]
    return _supervisor(llm, repository).graph.invoke({"messages": messages})


# --- Satu sub-tugas ----------------------------------------------------------


def test_single_sql_subtask_passes_result_through(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(("sql", SQL_Q)),
            sql={SQL_Q: COUNT_SQL},
            answers={SQL_Q: "Ada 3 lowongan."},
        )
    )
    repository = make_repository(CHUNKS)
    result = _invoke(llm, repository, SQL_Q)["result"]

    assert result.source == "sql"
    assert result.answer == "Ada 3 lowongan."
    assert result.rows == [{"jumlah": 3}]
    assert result.parts == []
    assert result.error is None
    assert repository.vector_db.calls == []


def test_usage_counts_planner_worker_and_sql_generation(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(("sql", SQL_Q)),
            sql={SQL_Q: COUNT_SQL},
            answers={SQL_Q: "Ada 3 lowongan."},
        )
    )
    usage = _invoke(llm, make_repository(), SQL_Q)["result"].usage

    assert sorted(call.step for call in usage.calls) == ["planner", "sql", "sql", "sql_generate"]
    assert usage.llm_calls == len(llm.received) == 4
    assert usage.total_tokens == 60


def test_single_rag_subtask_keeps_only_cited_candidates(make_repository):
    answer = "Ada:\n- [ID:1] Data Analyst - PT Alpha (Jakarta): cocok.\n- [ID:9] Palsu - X (Y): z."
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(("rag", RAG_Q)), answers={RAG_Q: answer}))
    result = _invoke(llm, make_repository(CHUNKS), RAG_Q)["result"]

    assert result.source == "rag"
    assert result.job_ids == [1]
    assert [row["id"] for row in result.rows] == [1]
    assert result.error == "ID di luar kandidat dibuang: [9]"


# --- Beberapa sub-tugas ------------------------------------------------------


def test_sql_and_rag_subtasks_are_combined_in_plan_order(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(("sql", SQL_Q), ("rag", RAG_Q)),
            sql={SQL_Q: COUNT_SQL},
            answers={SQL_Q: "Ada 3 lowongan.", RAG_Q: "- [ID:2] Data Analyst Intern: cocok."},
        )
    )
    state = _invoke(llm, make_repository(CHUNKS), f"{SQL_Q} {RAG_Q}")
    result = state["result"]

    assert result.source == "multi"
    assert [part.source for part in result.parts] == ["sql", "rag"]
    assert result.parts[0].answer == f"**{SQL_Q}**\nAda 3 lowongan."
    assert result.answer == "\n\n".join(part.answer for part in result.parts)
    assert result.job_ids == [2]
    assert [sub.source for sub in state["subresults"]] == ["sql", "rag"]
    assert result.usage.llm_calls == len(llm.received)


def test_two_sql_subtasks_run_separately(make_repository):
    other = "Berapa lowongan di Surabaya?"
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(("sql", SQL_Q), ("sql", other)),
            sql={
                SQL_Q: COUNT_SQL,
                other: COUNT_SQL.replace("jakarta", "surabaya"),
            },
            answers={SQL_Q: "Jakarta 3.", other: "Surabaya 1."},
        )
    )
    result = _invoke(llm, make_repository(), "Jakarta dan Surabaya?")["result"]

    assert [part.rows for part in result.parts] == [[{"jumlah": 3}], [{"jumlah": 1}]]


def test_more_than_three_subtasks_are_cut_to_three(make_repository):
    questions = [f"Pertanyaan {index}" for index in range(4)]
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(*[("rag", q) for q in questions]),
            answers={q: f"jawaban {q}" for q in questions},
        )
    )
    state = _invoke(llm, make_repository(CHUNKS), "empat pertanyaan")

    assert [task["question"] for task in state["subtasks"]] == questions[:3]
    assert state["skipped"] == questions[3:]
    assert state["result"].parts[-1].answer == f"{SUPERVISOR_SKIPPED_NOTE}\n- Pertanyaan 3"


def test_declined_part_adds_note_to_single_answer(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(("sql", SQL_Q), declined=["hapus data"]),
            sql={SQL_Q: COUNT_SQL},
            answers={SQL_Q: "Ada 3 lowongan."},
        )
    )
    result = _invoke(llm, make_repository(), f"{SQL_Q} lalu hapus data")["result"]

    assert result.source == "multi"
    assert [part.answer for part in result.parts] == ["Ada 3 lowongan.", SUPERVISOR_DECLINED_NOTE]


def test_skipped_parts_are_listed_in_note(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(("sql", SQL_Q), skipped=["Gaji di Bali?"]),
            sql={SQL_Q: COUNT_SQL},
            answers={SQL_Q: "Ada 3 lowongan."},
        )
    )
    result = _invoke(llm, make_repository(), "banyak pertanyaan")["result"]

    assert result.parts[-1].answer == f"{SUPERVISOR_SKIPPED_NOTE}\n- Gaji di Bali?"


# --- Tanpa worker ------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "message"),
    [("refuse", SUPERVISOR_REFUSE_MESSAGE), ("unsupported", SUPERVISOR_UNSUPPORTED_MESSAGE)],
)
def test_fixed_responses_skip_workers(make_repository, response, message):
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(response=response)))
    result = _invoke(llm, make_repository(), "pertanyaan")["result"]

    assert result.answer == message
    assert result.source == "general"
    assert len(llm.received) == 1


@pytest.mark.parametrize(
    ("response", "system"),
    [("out_of_scope", SUPERVISOR_CHAT_SYSTEM), ("clarify", SUPERVISOR_CLARIFY_SYSTEM)],
)
def test_chat_responses_use_llm_reply(make_repository, response, system):
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(response=response), chat="Halo juga."))
    result = _invoke(llm, make_repository(), "Halo")["result"]

    assert result.answer == "Halo juga."
    assert llm.received[-1][0].content == system
    assert result.usage.llm_calls == 2


def test_chat_reply_with_job_data_falls_back_to_fixed_message(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(plan=plan_reply(response="out_of_scope"), chat="Coba [ID:4] ya.")
    )
    assert _invoke(llm, make_repository(), "Halo")["result"].answer == SUPERVISOR_GENERAL_MESSAGE


def test_chat_llm_error_falls_back_to_fixed_message(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(plan=plan_reply(response="clarify"), chat=RuntimeError("API down"))
    )
    assert _invoke(llm, make_repository(), "Ada?")["result"].answer == SUPERVISOR_CLARIFY_MESSAGE


def test_answer_without_subtasks_becomes_clarify(make_repository):
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(), chat="Posisi apa?"))
    state = _invoke(llm, make_repository(), "Ada info kerjaan?")

    assert state["response"] == "clarify"
    assert state["result"].answer == "Posisi apa?"


def test_planner_error_becomes_clarify_without_crash(make_repository):
    llm = ScriptedLLM(reply=agent_script(plan=RuntimeError("API down"), chat="Posisi apa?"))
    state = _invoke(llm, make_repository(), "Lowongan?")

    assert state["response"] == "clarify"
    assert state["result"].answer == "Posisi apa?"


def test_unparsable_plan_becomes_clarify_and_keeps_usage(make_repository):
    llm = ScriptedLLM(reply=agent_script(plan=tool_call("Plan", response="answer"), chat="?"))
    state = _invoke(llm, make_repository(), "Lowongan?")

    assert state["response"] == "clarify"
    assert state["planner_usage"].llm_calls == 1


def test_planner_gets_labelled_history_and_last_question_in_one_message(make_repository):
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(response="refuse")))
    history = [
        {"role": "user", "content": "Lowongan marketing di Medan"},
        {"role": "assistant", "content": "Ada 2 lowongan."},
    ]
    _invoke(llm, make_repository(), "Yang full time?", history)

    planner_messages = llm.received[0]
    assert len(planner_messages) == 2
    assert planner_messages[1].content == SUPERVISOR_PLANNER_USER.format(
        history="Pengguna: Lowongan marketing di Medan\nAsisten: Ada 2 lowongan.",
        question="Yang full time?",
    )


def test_planner_without_history_says_so(make_repository):
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(response="refuse")))
    _invoke(llm, make_repository(), "Lowongan admin")

    assert llm.received[0][1].content == SUPERVISOR_PLANNER_USER.format(
        history=SUPERVISOR_NO_HISTORY, question="Lowongan admin"
    )


def test_sql_list_answer_is_completed_from_rows(make_repository):
    list_sql = "SELECT id, job_title, company_name, location FROM jobs WHERE id IN (1, 4)"
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(("sql", SQL_Q)),
            sql={SQL_Q: list_sql},
            answers={SQL_Q: "Berikut lowongannya:\n- [ID:1] Data Analyst - PT Alpha (Jakarta)"},
        )
    )
    result = _invoke(llm, make_repository(), SQL_Q)["result"]

    assert result.answer.endswith("- [ID:4] Marketing Manager - PT Delta (Jakarta Raya (Hibrid))")


def test_graph_structure(make_repository):
    llm = ScriptedLLM(reply=agent_script())
    nodes = set(_supervisor(llm, make_repository()).graph.get_graph().nodes)
    assert {"planner", "rag_agent", "sql_agent", "synthesize", "respond"} <= nodes


# --- Helper ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "valid", "invalid"),
    [
        ("[ID:1] dan [ID: 2]", [1, 2], []),
        ("[id:2] lalu [ID:9] lalu [ID:2]", [2], [9]),
        ("tanpa sitasi", [], []),
    ],
)
def test_check_citations_formats(answer, valid, invalid):
    assert check_citations(answer, [1, 2, 3]) == (valid, invalid)


def test_chat_context_splits_last_question_from_history():
    messages = [
        {"role": "user", "content": "Halo, saya Budi"},
        {"role": "assistant", "content": "Halo Budi."},
        {"role": "user", "content": "Siapa saya?"},
    ]
    question, history, previous = _chat_context(messages)

    assert question == "Siapa saya?"
    assert history == "Pengguna: Halo, saya Budi\nAsisten: Halo Budi."
    assert previous == "1. Halo, saya Budi"
