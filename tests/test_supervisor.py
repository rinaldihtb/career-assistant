"""Supervisor LangGraph (planner + fan-out paralel) dengan LLM palsu (gratis, tanpa API key).

Sub-tugas berjalan paralel, jadi urutan panggilan LLM antar sub-tugas tidak pasti.
Karena itu test memakai ``ScriptedLLM`` yang memilih balasan berdasarkan isi prompt
(pertanyaan sub-tugas), bukan berdasarkan urutan panggilan. Balasan planner berupa
``AIMessage`` dengan ``tool_calls``, persis bentuk yang dibaca parser structured
output LangChain.
"""

import hashlib
import json
import threading
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from agents import supervisor_agent
from agents.prompts import (
    CV_NO_MATCHES_MESSAGE,
    RAG_NO_CANDIDATE_MESSAGE,
    SQL_FAILED_MESSAGE,
    SUPERVISOR_CLARIFY_MESSAGE,
    SUPERVISOR_DECLINED_NOTE,
    SUPERVISOR_FAILED_MESSAGE,
    SUPERVISOR_GENERAL_MESSAGE,
    SUPERVISOR_NO_HISTORY,
    SUPERVISOR_REFUSE_MESSAGE,
    SUPERVISOR_SKIPPED_NOTE,
    SUPERVISOR_UNSUPPORTED_MESSAGE,
)
from agents.supervisor_agent import (
    MAX_HISTORY_CHARS,
    MAX_HISTORY_MESSAGES,
    MAX_PROFILE_SKILLS,
    SUPERVISOR_GRAPH,
    CVPlan,
    Plan,
    SupervisorContext,
    format_history,
    format_profile,
    planner_messages,
    run_supervisor,
)
from cv.match import JobMatch, MatchResult
from cv.profile import CVProfile, Experience

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
LONG = "Deskripsi pekerjaan yang cukup panjang untuk lolos batas chunk minimum."
COUNT_SQL = (
    "```sql\nSELECT COUNT(*) AS jumlah FROM jobs WHERE lower(location) LIKE '%jakarta%'\n```"
)


class ScriptedLLM(BaseChatModel):
    """LLM palsu: planner membalas ``plan``; agent dibalas sesuai kunci di prompt.

    Attributes:
        plan: Balasan planner (``AIMessage`` tool call) atau Exception.
        replies: ``{potongan_teks_prompt: [balasan, ...]}``. Panggilan agent mengambil
            balasan pertama dari kunci yang muncul di prompt user-nya. Satu sub-tugas
            selalu berjalan di satu thread, jadi urutan di dalam satu kunci tetap pasti.
        received: Semua daftar pesan yang diterima, untuk diperiksa test.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Any, bukan AIMessage | Exception: Pydantic akan mencoba mengubah Exception menjadi AIMessage.
    plan: Any = None
    replies: dict[str, list[Any]] = Field(default_factory=dict)
    received: list = Field(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        with self._lock:
            self.received.append(messages)
            if messages[0].content.startswith("Kamu planner"):
                reply = self.plan
            else:
                prompt = messages[-1].content
                key = next((k for k in self.replies if k in prompt and self.replies[k]), None)
                assert key is not None, f"tidak ada balasan untuk prompt: {prompt[:120]}"
                reply = self.replies[key].pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = reply if isinstance(reply, AIMessage) else AIMessage(reply, usage_metadata=USAGE)
        return ChatResult(generations=[ChatGeneration(message=message)])


def plan_reply(*subtasks, response="answer", skipped=(), declined=(), name="Plan") -> AIMessage:
    """Balasan planner palsu: tool call ``Plan`` (atau ``CVPlan`` saat CV aktif);
    ``subtasks`` = pasangan (agent, pertanyaan)."""
    args = {
        "response": response,
        "subtasks": [{"agent": agent, "question": question} for agent, question in subtasks],
        "skipped": list(skipped),
        "declined": list(declined),
    }
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": "1"}], usage_metadata=USAGE
    )


@pytest.fixture
def store(fake_vector_store):
    return fake_vector_store([(1, LONG, 0.70), (2, LONG, 0.60)])


def _run(question, llm, store, db, history=(), profile=None, matches=None):
    return run_supervisor(
        question,
        llm=llm,
        vector_store=store,
        db_path=db,
        history=history,
        profile=profile,
        matches=matches,
    )


def _state(question, llm, store, db, history=(), profile=None, matches=None):
    context = SupervisorContext(
        llm=llm, vector_store=store, db_path=db, profile=profile, matches=matches
    )
    return SUPERVISOR_GRAPH.invoke(
        {"question": question, "history": list(history)}, context=context
    )


# --- Satu sub-tugas: diteruskan apa adanya -----------------------------------


def test_single_sql_subtask_passes_result_through(jobs_db, store):
    q = "Berapa lowongan di Jakarta?"
    llm = ScriptedLLM(plan=plan_reply(("sql", q)), replies={q: [COUNT_SQL, "Ada 3 lowongan."]})
    result = _run(q, llm, store, jobs_db)

    assert result.source == "sql"
    assert result.answer == "Ada 3 lowongan."  # tidak dirangkum ulang
    assert result.rows == [{"jumlah": 3}]
    assert result.parts == []
    assert [call.step for call in result.usage.calls] == ["planner", "sql_generate", "sql_answer"]
    assert result.usage.total_tokens == 45
    assert result.error is None
    assert store.calls == []


def test_single_rag_subtask_keeps_citation_guard(jobs_db, store):
    q = "Lowongan data analyst"
    llm = ScriptedLLM(plan=plan_reply(("rag", q)), replies={q: ["Cocok: [ID:1] dan [ID:999]."]})
    result = _run(q, llm, store, jobs_db)

    assert result.source == "rag"
    assert result.job_ids == [1]
    assert result.error == "ID di luar kandidat dibuang: [999]"
    assert result.usage.llm_calls == 2


# --- Beberapa sub-tugas: paralel lalu digabung tanpa LLM ---------------------


def test_sql_and_rag_subtasks_are_combined_in_plan_order(jobs_db, store):
    q_sql, q_rag = "Berapa lowongan di Jakarta?", "Lowongan data analyst"
    llm = ScriptedLLM(
        plan=plan_reply(("sql", q_sql), ("rag", q_rag)),
        replies={q_sql: [COUNT_SQL, "Ada 3 lowongan."], q_rag: ["- [ID:1] Data Analyst: cocok."]},
    )
    state = _state("gabungan", llm, store, jobs_db)
    result = state["result"]

    assert result.source == "multi"
    assert [part.source for part in result.parts] == ["sql", "rag"]
    assert result.parts[0].answer == f"**{q_sql}**\nAda 3 lowongan."
    assert result.parts[1].answer == f"**{q_rag}**\n- [ID:1] Data Analyst: cocok."
    assert result.answer == "\n\n".join(part.answer for part in result.parts)
    assert result.job_ids == [1]
    steps = [call.step for call in result.usage.calls]
    assert steps == ["planner", "sql_generate", "sql_answer", "rag_answer"]  # urut sub-tugas
    assert result.usage.total_tokens == 60
    # Kontrak untuk evaluation.checks.extract_run.
    assert state["subtasks"] == [
        {"agent": "sql", "question": q_sql},
        {"agent": "rag", "question": q_rag},
    ]
    assert [sub.source for sub in state["subresults"]] == ["sql", "rag"]
    assert store.calls[0]["query"] == q_rag


def test_two_sql_subtasks_run_separately(jobs_db, store):
    q1, q2 = "Rata-rata gaji di Jakarta?", "Daftar lowongan di Jakarta"
    llm = ScriptedLLM(
        plan=plan_reply(("sql", q1), ("sql", q2)),
        replies={
            q1: [
                "```sql\nSELECT AVG(salary_min) AS rata FROM jobs\n```",
                "Rata-rata Rp10.000.000.",
            ],
            q2: [
                "```sql\nSELECT id, job_title FROM jobs WHERE id = 1\n```",
                "- [ID:1] Data Analyst",
            ],
        },
    )
    result = _run("gabungan", llm, store, jobs_db)

    assert [part.sql.splitlines()[0] for part in result.parts] == [
        "SELECT AVG(salary_min) AS rata FROM jobs",
        "SELECT id, job_title FROM jobs WHERE id = 1",
    ]
    assert result.usage.llm_calls == 5


def test_failed_subtask_does_not_break_others(jobs_db, store):
    q1, q2 = "Rata-rata gaji di Jakarta?", "Lowongan data analyst"
    llm = ScriptedLLM(
        plan=plan_reply(("sql", q1), ("rag", q2)),
        replies={q1: [ValueError("LLM mati")], q2: ["- [ID:2] Intern: cocok."]},
    )
    result = _run("gabungan", llm, store, jobs_db)

    assert result.parts[0].answer.endswith(SQL_FAILED_MESSAGE)
    assert result.job_ids == [2]
    assert "LLM error: LLM mati" in result.error


def test_more_than_three_subtasks_are_skipped_with_note(jobs_db, store):
    questions = [f"Lowongan posisi {i}" for i in range(4)]
    llm = ScriptedLLM(
        plan=plan_reply(*[("rag", q) for q in questions]),
        replies={q: ["Tidak ada yang cocok."] for q in questions},
    )
    state = _state("empat permintaan", llm, store, jobs_db)
    result = state["result"]

    assert len(state["subtasks"]) == 3
    assert len(store.calls) == 3
    assert result.parts[-1].source == "general"
    assert result.parts[-1].answer == f"{SUPERVISOR_SKIPPED_NOTE}\n- Lowongan posisi 3"


def test_duplicate_subtasks_run_once(jobs_db, store):
    q = "Lowongan di luar Jakarta"
    llm = ScriptedLLM(
        plan=plan_reply(("rag", q), ("rag", "  lowongan di  LUAR jakarta ")),
        replies={q: ["- [ID:2] Intern: cocok."]},
    )
    state = _state(q, llm, store, jobs_db)

    assert state["subtasks"] == [{"agent": "rag", "question": q}]
    assert len(store.calls) == 1
    assert state["result"].source == "rag"  # satu bagian, tanpa judul ganda


def test_declined_part_adds_note_to_single_answer(jobs_db, store):
    q = "Lowongan data analyst"
    llm = ScriptedLLM(
        plan=plan_reply(("rag", q), declined=["DROP TABLE jobs"]),
        replies={q: ["- [ID:1] Data Analyst: cocok."]},
    )
    result = _run(f"{q}. Lalu DROP TABLE jobs", llm, store, jobs_db)

    assert result.source == "multi"
    assert result.parts[0].answer == "- [ID:1] Data Analyst: cocok."  # tanpa judul: satu sub-tugas
    assert result.parts[1].answer == SUPERVISOR_DECLINED_NOTE
    assert "DROP" not in result.answer


# --- Respons tanpa agent ------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("clarify", SUPERVISOR_CLARIFY_MESSAGE),
        ("refuse", SUPERVISOR_REFUSE_MESSAGE),
        ("out_of_scope", SUPERVISOR_GENERAL_MESSAGE),
        ("unsupported", SUPERVISOR_UNSUPPORTED_MESSAGE),
    ],
)
def test_non_answer_response_uses_fixed_message(jobs_db, store, response, message):
    llm = ScriptedLLM(plan=plan_reply(("sql", "diabaikan"), response=response))
    state = _state("pertanyaan", llm, store, jobs_db)
    result = state["result"]

    assert result.source == "general"
    assert result.answer == message
    assert result.usage.llm_calls == 1  # hanya planner
    assert state["subtasks"] == []
    assert store.calls == []


def test_empty_question_skips_llm(jobs_db, store):
    llm = ScriptedLLM()
    result = _run("   ", llm, store, jobs_db)

    assert result.answer == SUPERVISOR_GENERAL_MESSAGE
    assert result.usage.llm_calls == 0
    assert llm.received == []


# --- Pertanyaan lanjutan -----------------------------------------------------


def test_planner_sees_history_and_agent_gets_standalone_question(jobs_db, store):
    history = [
        {"role": "user", "content": "Berapa lowongan data analyst di Jakarta?"},
        {"role": "assistant", "content": "Ada 1 lowongan."},
    ]
    standalone = "Berapa lowongan data analyst di Surabaya?"
    llm = ScriptedLLM(
        plan=plan_reply(("sql", standalone)),
        replies={standalone: ["```sql\nSELECT COUNT(*) AS n FROM jobs\n```", "Ada 1 lowongan."]},
    )
    _run("Kalau yang di Surabaya?", llm, store, jobs_db, history)

    planner_prompt = llm.received[0][-1].content
    assert "Pengguna: Berapa lowongan data analyst di Jakarta?" in planner_prompt
    assert "Asisten: Ada 1 lowongan." in planner_prompt
    assert "Pertanyaan terakhir pengguna: Kalau yang di Surabaya?" in planner_prompt
    assert f"Pertanyaan: {standalone}" in llm.received[1][-1].content


def test_format_history_keeps_job_ids_of_long_answers():
    answer = "Ada 4 lowongan:\n" + "\n".join(
        f"- [ID:{i}] Data Analyst - PT Nama Perusahaan Panjang {i} (Jakarta Raya): "
        + "alasan panjang " * 10
        for i in (10, 14, 28, 56)
    )
    line = format_history([{"role": "assistant", "content": answer}])

    assert line == (
        "Asisten: Ada 4 lowongan: "
        "- [ID:10] Data Analyst - PT Nama Perusahaan Panjang 10 (Jakarta Raya) "
        "- [ID:14] Data Analyst - PT Nama Perusahaan Panjang 14 (Jakarta Raya) "
        "- [ID:28] Data Analyst - PT Nama Perusahaan Panjang 28 (Jakarta Raya) "
        "- [ID:56] Data Analyst - PT Nama Perusahaan Panjang 56 (Jakarta Raya)"
    )


def test_format_history_keeps_recent_and_truncates():
    history = [{"role": "user", "content": f"pesan {i}"} for i in range(10)]
    history.append({"role": "user", "content": "x" * (MAX_HISTORY_CHARS + 50)})
    lines = format_history(history).splitlines()

    assert len(lines) == MAX_HISTORY_MESSAGES
    assert lines[0] == "Pengguna: pesan 5"
    assert lines[-1] == "Pengguna: " + "x" * MAX_HISTORY_CHARS + "..."


def test_format_history_empty():
    assert format_history([]) == SUPERVISOR_NO_HISTORY


# --- Planner gagal -> fallback RAG, tidak pernah melempar exception ----------


def test_planner_error_falls_back_to_rag(jobs_db, store):
    q = "Lowongan data analyst"
    llm = ScriptedLLM(plan=TimeoutError("rate limit"), replies={q: ["Tidak ada yang cocok."]})
    result = _run(q, llm, store, jobs_db)

    assert result.source == "rag"
    assert "Planner error: rate limit" in result.error
    assert store.calls[0]["query"] == q


def test_invalid_planner_output_falls_back_to_rag(jobs_db, fake_vector_store):
    llm = ScriptedLLM(plan=plan_reply(("cuaca", "Lowongan pilot")))
    store = fake_vector_store([(1, LONG, 0.30)])  # di bawah threshold: 0 LLM call RAG
    result = _run("Lowongan pilot", llm, store, jobs_db)

    assert result.source == "rag"
    assert result.answer == RAG_NO_CANDIDATE_MESSAGE
    assert result.error.startswith("Planner output tidak valid")
    assert result.usage.llm_calls == 1  # panggilan planner tetap dihitung


def test_answer_without_subtasks_falls_back_to_rag(jobs_db, fake_vector_store):
    llm = ScriptedLLM(plan=plan_reply())
    store = fake_vector_store([(1, LONG, 0.30)])
    result = _run("Lowongan pilot", llm, store, jobs_db)

    assert result.error == "Planner tidak memberi sub-tugas"
    assert store.calls[0]["query"] == "Lowongan pilot"


def test_planner_and_agent_errors_are_both_kept(jobs_db, store):
    q = "Lowongan data analyst"
    llm = ScriptedLLM(plan=plan_reply(("nope", "x")), replies={q: [ValueError("LLM mati")]})
    result = _run(q, llm, store, jobs_db)

    assert result.error.startswith("Planner output tidak valid")
    assert "LLM error: LLM mati" in result.error


def test_unexpected_graph_error_returns_fixed_message(jobs_db, store, monkeypatch):
    class BrokenGraph:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("graph rusak")

    monkeypatch.setattr(supervisor_agent, "SUPERVISOR_GRAPH", BrokenGraph())
    result = _run("Halo", ScriptedLLM(), store, jobs_db)

    assert result.answer == SUPERVISOR_FAILED_MESSAGE
    assert result.error == "Supervisor error: graph rusak"


# --- Struktur graph ----------------------------------------------------------


def test_graph_structure():
    graph = SUPERVISOR_GRAPH.get_graph()
    nodes = set(graph.nodes) - {"__start__", "__end__"}
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert nodes == {"planner", "sql_agent", "rag_agent", "cv_agent", "synthesize", "respond"}
    assert {
        ("planner", "sql_agent"),
        ("planner", "rag_agent"),
        ("planner", "cv_agent"),
        ("planner", "respond"),
    } <= edges
    assert {
        ("sql_agent", "synthesize"),
        ("rag_agent", "synthesize"),
        ("cv_agent", "synthesize"),
    } <= edges
    assert {("synthesize", "__end__"), ("respond", "__end__")} <= edges


# --- CV aktif ----------------------------------------------------------------

# sha256 prompt dan skema planner tanpa CV, sama dengan versi yang dinilai set eval 51 kasus
# (tests/eval/queries.yaml). Jika test ini gagal, prompt tanpa CV berubah dan `make eval`
# wajib dijalankan ulang sebelum angka lama dipakai.
PLANNER_SYSTEM_SHA256 = "93d1df89e149a7239fa3d312bc645f49aaa46f991675752b252a2c2f6fcd6628"
PLANNER_USER_SHA256 = "79c091499bf9b419ac5fdc9db9682b9c6902c95b0cb0daf8ae2393ce8ce0b40b"
PLAN_SCHEMA_SHA256 = "597ab5da066be752ece658353942e407c7cf100058977a6cf9ce4f50268beebd"

PROFILE = CVProfile(
    full_name="Nadia Putri",
    target_role="Data Analyst",
    location="Tegalwangi Utara",
    education_level="S1",
    education_major="Statistika",
    skills=["SQL", "Excel"],
    experience_months=14,
    experiences=[
        Experience(title="Analis Data", company="PT Rahasia Sekali", start="2024-01", end="2025-03")
    ],
)


def cv_job(job_id: int, score: float, matched=(), missing=()) -> JobMatch:
    return JobMatch(
        job_id=job_id,
        job_title=f"Data Analyst {job_id}",
        company_name=f"PT {job_id}",
        location="Jakarta Raya",
        score=score,
        semantic=0.6,
        semantic_scaled=0.6,
        skill_portion=0.5,
        matched_skills=list(matched),
        missing_skills=list(missing),
    )


MATCHES = MatchResult(
    matches=[cv_job(5, 0.82, ["SQL"], ["Power BI"]), cv_job(3, 0.71, ["Excel"], ["Power BI"])],
    embedding_calls=1,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_planner_without_cv_is_unchanged():
    system, user = planner_messages("Q", [], None)

    assert _sha256(system.content) == PLANNER_SYSTEM_SHA256
    assert _sha256(user.content) == PLANNER_USER_SHA256
    schema = json.dumps(Plan.model_json_schema(), sort_keys=True)
    assert _sha256(schema) == PLAN_SCHEMA_SHA256
    assert "cv_match" not in system.content
    assert "<profil>" not in user.content


def test_planner_with_cv_gets_short_profile_without_personal_data(jobs_db, store):
    llm = ScriptedLLM(plan=plan_reply(("cv_match", "Lowongan cocok"), name="CVPlan"))
    result = _run("Lowongan yang cocok untukku?", llm, store, jobs_db, profile=PROFILE)

    system, user = (message.content for message in llm.received[0])
    assert '"cv_match"' in system
    assert '"cv_gap"' in system
    assert user.startswith(
        "<profil>\nPosisi tujuan: Data Analyst\nPengalaman: 1-3 tahun\n"
        "Pendidikan: S1 Statistika\nSkill: SQL, Excel\n</profil>"
    )
    assert user.endswith("Pertanyaan terakhir pengguna: Lowongan yang cocok untukku?")
    for private in ("Nadia", "Tegalwangi", "PT Rahasia", "Analis Data"):
        assert private not in system + user
    assert result.answer == CV_NO_MATCHES_MESSAGE  # matches belum ada


def test_cv_match_is_answered_from_matches_with_one_llm_call(jobs_db, store):
    q = "Lowongan yang paling cocok dengan CV-ku"
    llm = ScriptedLLM(plan=plan_reply(("cv_match", q), name="CVPlan"))
    state = _state("Ada lowongan buat aku?", llm, store, jobs_db, profile=PROFILE, matches=MATCHES)
    result = state["result"]

    assert result.source == "cv"
    assert result.job_ids == [5, 3]
    assert "- [ID:5] Data Analyst 5 - PT 5 (Jakarta Raya): 82% cocok" in result.answer
    assert [call.step for call in result.usage.calls] == ["planner"]
    assert state["subtasks"] == [{"agent": "cv_match", "question": q}]
    assert store.calls == []


def test_cv_gap_and_sql_are_combined(jobs_db, store):
    q_gap, q_sql = "Skill yang perlu kupelajari dari CV-ku", "Berapa lowongan di Jakarta?"
    llm = ScriptedLLM(
        plan=plan_reply(("cv_gap", q_gap), ("sql", q_sql), name="CVPlan"),
        replies={q_sql: [COUNT_SQL, "Ada 3 lowongan."]},
    )
    result = _run("gabungan", llm, store, jobs_db, profile=PROFILE, matches=MATCHES)

    assert result.source == "multi"
    assert [part.source for part in result.parts] == ["cv", "sql"]
    assert result.parts[0].answer.startswith(f"**{q_gap}**\nDari 2 lowongan")
    assert "- Power BI: 2 dari 2 lowongan" in result.parts[0].answer
    assert [call.step for call in result.usage.calls] == ["planner", "sql_generate", "sql_answer"]


def test_duplicate_cv_subtasks_run_once(jobs_db, store):
    llm = ScriptedLLM(
        plan=plan_reply(
            ("cv_match", "Lowongan cocok untukku"),
            ("cv_match", "Lowongan yang sesuai CV-ku"),
            name="CVPlan",
        )
    )
    state = _state("x", llm, store, jobs_db, profile=PROFILE, matches=MATCHES)

    assert state["subtasks"] == [{"agent": "cv_match", "question": "Lowongan cocok untukku"}]
    assert state["result"].source == "cv"


def test_cv_agent_is_not_available_without_cv(jobs_db, fake_vector_store):
    llm = ScriptedLLM(plan=plan_reply(("cv_match", "Lowongan cocok untukku")))
    store = fake_vector_store([(1, LONG, 0.30)])
    result = _run("Lowongan cocok untukku", llm, store, jobs_db)

    assert result.source == "rag"  # skema Plan menolak cv_match -> fallback RAG
    assert result.error.startswith("Planner output tidak valid")


def test_cv_planner_schema_only_adds_cv_agents():
    plan = Plan.model_json_schema()
    cv_plan = CVPlan.model_json_schema()

    assert plan["$defs"]["SubTask"]["properties"]["agent"]["enum"] == ["sql", "rag"]
    assert cv_plan["$defs"]["CVSubTask"]["properties"]["agent"]["enum"] == [
        "sql",
        "rag",
        "cv_match",
        "cv_gap",
    ]
    assert plan["required"] == cv_plan["required"]


def test_format_profile_limits_skills_and_skips_empty_lines():
    skills = [f"Skill {i}" for i in range(MAX_PROFILE_SKILLS + 5)]
    text = format_profile(CVProfile(skills=skills))

    assert text.splitlines() == [
        "Pengalaman: Tidak diketahui",
        "Skill: " + ", ".join(skills[:MAX_PROFILE_SKILLS]),
    ]


def test_format_history_keeps_job_ids_of_cv_answer():
    answer = run_supervisor(
        "x",
        llm=ScriptedLLM(plan=plan_reply(("cv_match", "x"), name="CVPlan")),
        vector_store=None,
        db_path="tidak-dipakai.sqlite",
        profile=PROFILE,
        matches=MATCHES,
    ).answer
    line = format_history([{"role": "assistant", "content": answer}])

    assert "- [ID:5] Data Analyst 5 - PT 5 (Jakarta Raya)" in line
    assert "- [ID:3] Data Analyst 3 - PT 3 (Jakarta Raya)" in line
