"""SupervisorAgent (planner + Send paralel + synthesize/respond) dengan LLM palsu.

Sub-tugas berjalan paralel, jadi ``ScriptedLLM`` memilih balasan dari isi prompt
(jenis prompt dan pertanyaan sub-tugas), bukan dari urutan panggilan. Balasan
planner berupa tool call ``Plan``, bentuk yang dibaca parser structured output.
"""

import hashlib

import pytest
from conftest import ScriptedLLM, agent_script, plan_reply, tool_call

from src.agents.prompts import (
    CV_ROLE_FAILED_MESSAGE,
    DATA_SOURCE_MESSAGE,
    JOB_DETAIL_MISSING,
    SUPERVISOR_CHAT_CV_NOTE,
    SUPERVISOR_CHAT_SYSTEM,
    SUPERVISOR_CLARIFY_MESSAGE,
    SUPERVISOR_CLARIFY_SYSTEM,
    SUPERVISOR_DECLINED_NOTE,
    SUPERVISOR_GENERAL_CV_MESSAGE,
    SUPERVISOR_GENERAL_MESSAGE,
    SUPERVISOR_NO_HISTORY,
    SUPERVISOR_PLANNER_CV_AGENTS,
    SUPERVISOR_PLANNER_USER,
    SUPERVISOR_REFUSE_MESSAGE,
    SUPERVISOR_SKIPPED_NOTE,
    SUPERVISOR_UNSUPPORTED_MESSAGE,
)
from src.agents.rag_agent import RagAgent
from src.agents.sql_agent import SqlAgent
from src.agents.supervisor_agent_graph import (
    CVPlan,
    Plan,
    SupervisorAgent,
    _chat_context,
    career_move_role,
    check_citations,
    core_role,
    format_profile,
    is_new_role,
    planner_messages,
    replace_self_references,
    role_from_question,
    target_roles,
)
from src.cv.match import JobMatch, MatchResult
from src.cv.profile import CVProfile

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


def _invoke(llm, repository, question, history=(), **cv):
    messages = [*history, {"role": "user", "content": question}]
    return _supervisor(llm, repository).graph.invoke({"messages": messages, **cv})


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


# --- CV aktif ----------------------------------------------------------------

PROFILE = CVProfile(
    full_name="Nadia Putri",
    target_role="Senior Data Analyst",
    location="Pekanbaru",
    education_level="S1",
    education_major="Statistika",
    skills=["SQL", "Excel"],
    experience_months=14,
    contacts=["0812-3456-7890", "nadia@mail.com"],
)
MATCHES = MatchResult(
    matches=[
        JobMatch(
            job_id=job_id,
            job_title="Data Analyst",
            score=0.9 - job_id / 100,
            semantic=0.6,
            semantic_scaled=0.6,
            matched_skills=["SQL"],
            missing_skills=["Python"],
        )
        for job_id in range(1, 8)
    ]
)
CV = {"profile": PROFILE, "matches": MATCHES}


def test_planner_prompt_without_cv_is_unchanged():
    """Hash prompt planner tanpa CV yang diukur dengan set eval 51 kasus; berubah = ukur ulang."""
    system, user = planner_messages("Lowongan admin", SUPERVISOR_NO_HISTORY, None)

    assert hashlib.sha256(system.content.encode()).hexdigest()[:12] == "3ae93e488122"
    assert "cv_match" not in system.content
    assert user.content.startswith("Riwayat percakapan")


def test_planner_with_cv_gets_cv_workers_and_profile_without_identity(make_repository):
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(response="refuse", schema="CVPlan")))
    _invoke(llm, make_repository(), "Lowongan yang cocok untukku", **CV)

    system, user = llm.received[0]
    assert SUPERVISOR_PLANNER_CV_AGENTS in system.content
    assert format_profile(PROFILE) not in system.content
    assert user.content.startswith(f"<profil>\n{format_profile(PROFILE)}\n</profil>")
    for private in ["Nadia", "Pekanbaru", *PROFILE.contacts]:
        assert private not in system.content + user.content


def test_cv_plan_cannot_choose_clarify():
    def response_enum(schema: type) -> list[str]:
        return schema.model_json_schema()["properties"]["response"]["enum"]

    assert "clarify" in response_enum(Plan)
    assert "clarify" not in response_enum(CVPlan)
    assert set(response_enum(CVPlan)) == set(response_enum(Plan)) - {"clarify"}


def test_cv_subtasks_run_without_extra_llm_call(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(
                ("cv_match", "Lowongan yang paling cocok dengan CV-ku"),
                ("cv_gap", "Skill yang perlu kupelajari dari CV-ku"),
                schema="CVPlan",
            )
        )
    )
    state = _invoke(llm, make_repository(), "Lowongan buatku dan skill yang kurang?", **CV)
    result = state["result"]

    assert [part.source for part in result.parts] == ["cv", "cv"]
    assert result.job_ids == [1, 2, 3, 4, 5]
    assert "- Python: 7 dari 7 lowongan" in result.parts[1].answer
    assert result.usage.llm_calls == 1


def test_cv_subtask_without_matches_points_to_cv_page(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(plan=plan_reply(("cv_match", "Lowongan buatku"), schema="CVPlan"))
    )
    state = _invoke(llm, make_repository(), "Lowongan buatku", profile=PROFILE, matches=None)

    assert state["result"].source == "cv"
    assert state["result"].answer.startswith("Belum ada hasil Cocokkan CV")


def test_out_of_scope_with_cv_mentions_cv_and_falls_back_to_cv_message(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(response="out_of_scope", schema="CVPlan"),
            chat=RuntimeError("provider error"),
        )
    )
    state = _invoke(llm, make_repository(), "Kamu bisa apa?", **CV)

    assert state["result"].answer == SUPERVISOR_GENERAL_CV_MESSAGE
    assert llm.received[1][0].content.endswith(SUPERVISOR_CHAT_CV_NOTE)


def test_format_profile_sends_core_role_experience_education_skills():
    assert format_profile(PROFILE) == (
        "Posisi tujuan: Data Analyst\nPengalaman: 1-3 tahun\nPendidikan: S1 Statistika\n"
        "Skill: SQL, Excel"
    )
    assert format_profile(CVProfile()) == "Pengalaman: Tidak diketahui"


@pytest.mark.parametrize(
    ("role", "core"),
    [
        ("Senior Teknisi Listrik", "Teknisi Listrik"),
        ("Sales & Marketing Executive", "Sales & Marketing"),
        ("Account Executive", "Account Executive"),
        ("HR Officer", "HR Officer"),
        ("Jr. Data Analyst", "Data Analyst"),
    ],
)
def test_core_role_drops_job_level(role, core):
    assert core_role(role) == core


def test_target_roles_split_and_deduplicate():
    assert target_roles("Fullstack Developer dan Data Analyst / data analyst") == [
        "Fullstack Developer",
        "Data Analyst",
    ]
    assert target_roles(None) == []


# --- Penjaga rute ------------------------------------------------------------


def test_rag_subtask_with_salary_number_goes_to_sql(make_repository):
    question = "Lowongan data analyst di Jakarta dengan range gaji 5 juta sampai 8 juta"
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(("rag", question)),
            sql={question: COUNT_SQL},
            answers={question: "Ada 3 lowongan."},
        )
    )
    state = _invoke(llm, make_repository(), question)

    assert state["subtasks"] == [{"agent": "sql", "question": question}]
    assert state["result"].source == "sql"


def test_cv_match_with_city_becomes_rag_with_target_role(make_repository):
    rag_question = "Yang cocok untuk Data Analyst di Jakarta ada?"
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(("cv_match", "Lowongan yang cocok dengan CV-ku"), schema="CVPlan"),
            answers={rag_question: "Tidak ada."},
        )
    )
    state = _invoke(llm, make_repository(CHUNKS), "Yang cocok untukku di Jakarta ada?", **CV)

    assert state["subtasks"] == [{"agent": "rag", "question": rag_question}]


@pytest.mark.parametrize(
    "question",
    ["Lowongan yang cocok untukku", "Lowongan data analyst di Jakarta"],
)
def test_cv_match_stays_without_city_or_self_reference(make_repository, question):
    llm = ScriptedLLM(
        reply=agent_script(plan=plan_reply(("cv_match", "Lowongan buatku"), schema="CVPlan"))
    )
    state = _invoke(llm, make_repository(), question, **CV)

    assert [task["agent"] for task in state["subtasks"]] == ["cv_match"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Lowongan yang sesuai CV-ku di Surabaya", "Lowongan yang sesuai untuk HR di Surabaya"),
        ("Ada lowongan buat aku di Bandung?", "Ada lowongan untuk HR di Bandung?"),
        ("Kerjaan buatku di Medan", "Kerjaan untuk HR di Medan"),
        ("Gaji posisi incaranku berapa?", "Gaji HR berapa?"),
        ("Lowongan cocok dengan profil saya", "Lowongan cocok untuk HR"),
    ],
)
def test_replace_self_references(question, expected):
    assert replace_self_references(question, "HR") == expected
    assert replace_self_references(question, "") == question


# --- cv_role dan penjaga cv_match ------------------------------------------------


def test_cv_role_matches_again_for_the_named_role(make_repository, monkeypatch):
    calls = []

    def fake_match_role(self, role, profile):
        calls.append((role, profile))
        return MatchResult(matches=MATCHES.matches, embedding_calls=1)

    monkeypatch.setattr(SupervisorAgent, "match_role", fake_match_role)
    llm = ScriptedLLM(
        reply=agent_script(plan=plan_reply(("cv_role", "Product Manager"), schema="CVPlan"))
    )
    state = _invoke(llm, make_repository(), "Mau jadi product manager, perlu apa?", **CV)

    assert calls == [("Product Manager", PROFILE)]
    result = state["result"]
    assert result.source == "cv"
    assert result.answer.startswith("Untuk posisi Product Manager")
    assert result.job_ids == [match.job_id for match in MATCHES.matches][:5]


def test_cv_role_without_embedding_model_says_search_failed(make_repository):
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(("cv_role", "Pilot"), schema="CVPlan")))
    state = _invoke(llm, make_repository(), "Mau jadi pilot", **CV)

    assert state["result"].answer == CV_ROLE_FAILED_MESSAGE.format(role="Pilot")


@pytest.mark.parametrize(
    ("question", "agents"),
    [
        ("Carikan saya lowongan akuntan", ["rag"]),
        ("Lowongan yang cocok buatku dan lowongan akuntan", ["cv_match", "rag"]),
    ],
)
def test_cv_match_next_to_rag_needs_match_request(make_repository, question, agents):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(
                ("cv_match", "Lowongan yang paling cocok dengan CV-ku"),
                ("rag", "Lowongan akuntan"),
                schema="CVPlan",
            ),
            answers={"Lowongan akuntan": "Tidak ada."},
        )
    )
    state = _invoke(llm, make_repository(CHUNKS), question, **CV)

    assert [task["agent"] for task in state["subtasks"]] == agents


@pytest.mark.parametrize(
    ("question", "role"),
    [
        ("Skill yang perlu kupelajari untuk menjadi Product Manager", "Product Manager"),
        ("Aku mau pindah karier ke bidang keuangan.", "keuangan"),
        ("Project Manager", "Project Manager"),
        ("Skill untuk beralih karir ke bidang lain.", "lain"),
    ],
)
def test_role_from_question(question, role):
    assert role_from_question(question) == role


@pytest.mark.parametrize(
    ("role", "new"),
    [("Product Manager", True), ("Data Analyst", False), ("bidang lain", False), ("lain", False)],
)
def test_is_new_role_rejects_generic_and_own_role(role, new):
    assert is_new_role(role, PROFILE) is new  # PROFILE: Senior Data Analyst


@pytest.mark.parametrize(
    ("extra", "kept"),
    [
        (("cv_gap", "Skill yang perlu kupelajari dari CV-ku"), []),
        (("rag", "Lowongan Product Manager yang cocok untukku"), []),
        (("rag", "Lowongan Product Manager di Jakarta"), ["rag"]),
        (("sql", "Berapa rata-rata gaji Product Manager?"), ["sql"]),
    ],
)
def test_cv_role_drops_repeating_subtasks(make_repository, monkeypatch, extra, kept):
    monkeypatch.setattr(SupervisorAgent, "match_role", lambda self, role, profile: MATCHES)
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(
                ("cv_role", "Skill yang perlu kupelajari untuk menjadi Product Manager"),
                extra,
                schema="CVPlan",
            ),
            sql={extra[1]: COUNT_SQL},
            answers={extra[1]: "Tidak ada."},
        )
    )
    state = _invoke(llm, make_repository(CHUNKS), "Mau jadi product manager", **CV)

    assert state["subtasks"][0] == {"agent": "cv_role", "question": "Product Manager"}
    assert [task["agent"] for task in state["subtasks"][1:]] == kept


def test_cv_role_for_own_role_becomes_out_of_scope(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(
                ("cv_role", "Skill yang perlu kupelajari untuk menjadi Data Analyst"),
                schema="CVPlan",
            ),
            chat="Mau pindah ke bidang apa?",
        )
    )
    state = _invoke(llm, make_repository(), "bagaimana cara switch karir", **CV)

    assert state["response"] == "out_of_scope"
    assert state["subtasks"] == []
    assert state["result"].answer == "Mau pindah ke bidang apa?"


@pytest.mark.parametrize(
    ("second", "agents"),
    [
        (("rag", "Lowongan Data Analyst di Jakarta"), ["rag"]),
        (("rag", "Lowongan barista di Jakarta"), ["cv_match", "rag"]),
        (("sql", "Berapa jumlah lowongan Data Analyst di Jakarta?"), ["cv_match", "sql"]),
    ],
)
def test_cv_match_dropped_next_to_personal_rag_with_city(make_repository, second, agents):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(
                ("cv_match", "Lowongan yang paling cocok dengan CV-ku"), second, schema="CVPlan"
            ),
            sql={second[1]: COUNT_SQL},
            answers={second[1]: "Tidak ada."},
        )
    )
    state = _invoke(llm, make_repository(CHUNKS), "Lowongan yang sesuai CV-ku di Jakarta", **CV)

    assert [task["agent"] for task in state["subtasks"]] == agents


def test_career_switch_without_role_becomes_out_of_scope(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(
            plan=plan_reply(
                ("cv_match", "Lowongan yang paling cocok dengan CV-ku"), schema="CVPlan"
            ),
            chat="Mau pindah ke bidang apa?",
        )
    )
    state = _invoke(llm, make_repository(), "bagaimana cara saya untuk switch karir", **CV)

    assert state["response"] == "out_of_scope"
    assert state["result"].answer == "Mau pindah ke bidang apa?"


@pytest.mark.parametrize(
    ("question", "role"),
    [
        ("switch ke finance", "finance"),
        ("Pengen pindah karier ke bidang perhotelan, gimana?", "perhotelan"),
        ("bagaimana cara saya untuk switch karir", ""),
        ("pindah ke bidang lain", ""),
        ("mau pindah jadi data analyst", ""),
        ("Halo, jadi aku mau tanya", ""),
    ],
)
def test_career_move_role(question, role):
    assert career_move_role(question, PROFILE) == role  # PROFILE: Senior Data Analyst


def test_out_of_scope_career_move_with_role_becomes_cv_role(make_repository):
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(response="out_of_scope", schema="CVPlan")))
    state = _invoke(llm, make_repository(), "switch ke finance", **CV)

    assert state["response"] == "answer"
    assert state["subtasks"] == [{"agent": "cv_role", "question": "finance"}]
    assert state["result"].answer == CV_ROLE_FAILED_MESSAGE.format(role="finance")
    assert len(llm.received) == 1


def test_out_of_scope_without_cv_is_not_changed_by_career_move(make_repository):
    llm = ScriptedLLM(
        reply=agent_script(plan=plan_reply(response="out_of_scope"), chat="Mau ke bidang apa?")
    )
    state = _invoke(llm, make_repository(), "switch ke finance")

    assert state["response"] == "out_of_scope"
    assert state["result"].answer == "Mau ke bidang apa?"


# --- Rute data tanpa planner -----------------------------------------------------

MARKET_Q = "skill apa saja yang paling banyak dicari oleh perusahaan atau recruiter?"
SOURCE_Q = "berikan referensi darimana informasi tersebut anda dapatkan."


def _no_llm(messages):
    raise AssertionError("LLM tidak boleh dipanggil")


@pytest.mark.parametrize("cv, have", [(CV, True), ({}, False)])
def test_market_skill_question_counts_all_jobs_without_llm(
    make_repository, jobs_db, monkeypatch, cv, have
):
    monkeypatch.setenv("DB_SQLITE_PATH", str(jobs_db))
    llm = ScriptedLLM(reply=_no_llm)
    state = _invoke(llm, make_repository(), MARKET_Q, **cv)

    result = state["result"]
    assert llm.received == []
    assert state["subtasks"] == [{"agent": "skill_stats", "question": MARKET_Q}]
    assert result.source == "data"
    assert result.usage.llm_calls == 0
    assert "di 6 lowongan data kami" in result.answer
    assert "- Python: 2 lowongan (contoh ID 1, 5)" in result.answer
    assert ("Yang sudah ada di CV-mu: SQL." in result.answer) is have


@pytest.mark.parametrize(
    "question",
    [
        "skill apa yang paling dicari di Surabaya?",
        "Skill apa yang paling sering diminta tapi belum saya punya?",
        "skill apa yang paling dicari untuk data analyst?",
    ],
)
def test_market_skill_with_place_self_or_role_goes_to_planner(make_repository, question):
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(("cv_gap", "Skill CV"), schema="CVPlan")))
    state = _invoke(llm, make_repository(), question, **CV)

    assert llm.received
    assert [task["agent"] for task in state["subtasks"]] == ["cv_gap"]


def test_source_follow_up_explains_data_without_llm(make_repository, jobs_db, monkeypatch):
    monkeypatch.setenv("DB_SQLITE_PATH", str(jobs_db))
    llm = ScriptedLLM(reply=_no_llm)
    history = [
        {"role": "user", "content": MARKET_Q},
        {"role": "assistant", "content": "- Python: 2 lowongan (contoh ID 1, 5)"},
    ]
    state = _invoke(llm, make_repository(), SOURCE_Q, history, **CV)

    assert llm.received == []
    assert state["result"].answer == DATA_SOURCE_MESSAGE.format(total=6)


@pytest.mark.parametrize(
    "question, history",
    [
        (SOURCE_Q, ()),
        ("Lowongan HRD sumber daya manusia di Surabaya", ({"role": "user", "content": "Halo"},)),
    ],
)
def test_source_pattern_needs_history_and_skips_human_resources(make_repository, question, history):
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(response="out_of_scope"), chat="Halo."))
    state = _invoke(llm, make_repository(), question, history)

    assert llm.received
    assert state.get("subtasks", []) == []


@pytest.mark.parametrize("cv", [{}, CV])
def test_job_detail_by_id_reads_sqlite_without_llm(make_repository, jobs_db, monkeypatch, cv):
    monkeypatch.setenv("DB_SQLITE_PATH", str(jobs_db))
    llm = ScriptedLLM(reply=_no_llm)
    state = _invoke(llm, make_repository(), "Tampilkan deskripsi lengkap lowongan ID 4", **cv)

    result = state["result"]
    assert llm.received == []
    assert state["subtasks"] == [{"agent": "job_detail", "question": "4"}]
    assert (result.source, result.job_ids, result.usage.llm_calls) == ("sql", [4], 0)
    assert "- [ID:4] Marketing Manager - PT Delta (Jakarta Raya (Hibrid)): " in result.answer


def test_job_detail_for_unknown_id_says_it_is_missing(make_repository, jobs_db, monkeypatch):
    monkeypatch.setenv("DB_SQLITE_PATH", str(jobs_db))
    state = _invoke(ScriptedLLM(reply=_no_llm), make_repository(), "detail lowongan id 99")

    assert state["result"].answer == JOB_DETAIL_MISSING.format(id=99)
    assert state["result"].job_ids == []


def test_job_detail_with_two_ids_goes_to_planner(make_repository):
    question = "Bandingkan detail lowongan ID 1 dan ID 2"
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(response="out_of_scope"), chat="Halo."))
    state = _invoke(llm, make_repository(), question)

    assert llm.received
    assert state.get("subtasks", []) == []


@pytest.mark.parametrize("cv", [{}, CV])
def test_exact_job_title_searches_that_title_without_planner(make_repository, cv):
    question = "Lowongan Data Analyst Intern"
    llm = ScriptedLLM(reply=agent_script(answers={question: "Tidak ada yang cocok."}))
    state = _invoke(llm, make_repository(), "  data analyst   INTERN? ", **cv)

    assert state["subtasks"] == [{"agent": "rag", "question": question}]
    assert state["planner_usage"].llm_calls == 0
