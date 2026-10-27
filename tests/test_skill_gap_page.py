"""Halaman Skill Gap dengan Streamlit AppTest (tanpa browser dan tanpa API key).

``generate_learning_plan`` dan ``load_dependencies`` diganti fungsi palsu: yang dites adalah UI,
yaitu ajakan tanpa CV, bar "x dari N", satu panggilan LLM per versi profil, dan cache.
Logika agregasi dan rencana dites di test_cv_gap.py dan test_cv_plan.py.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_assistant_page import _use

from src.components import skill_gap
from src.components.cv_match import CVDependencies
from src.components.skill_gap import (
    ASSISTANT_QUESTION,
    DEPENDENCY_ERROR,
    NO_CV,
    NO_GAP,
    NO_MATCHES,
    PAGE_TITLE,
    PLAN_FAILED,
    PLAN_STATE_KEY,
    cached_plan,
    count_label,
    gap_caption,
    plan_usage_caption,
    week_label,
)
from src.cv.gap import SkillCount, compute_skill_gap
from src.cv.ingest import CV_STATE_KEY
from src.cv.match import JobMatch, MatchResult
from src.cv.plan import LearningPlan, LearningPlanResult, PlanWeek
from src.cv.profile import CVProfile
from src.models.agent import AgentResult, LLMUsage

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"
# Run pertama di satu proses pytest memuat pandas/pyarrow: lebih dari 10 detik di laptop macOS.
APP_TIMEOUT = 30
DEPS = CVDependencies(llm="llm", embeddings="emb", vector_store="store", db_path="db.sqlite")
PROFILE = CVProfile(target_role="Data Analyst", skills=["SQL", "Excel"], experience_months=14)
PLAN = LearningPlan(
    weeks=[
        PlanWeek(title=f"Fokus {n}", skills=["Power BI"], tasks=[f"Tugas {n}a", f"Tugas {n}b"])
        for n in range(1, 5)
    ]
)
PLAN_USAGE = LLMUsage(llm_calls=1, input_tokens=700, output_tokens=340, total_tokens=1040)


def job(job_id: int, score: float, matched: list[str], missing: list[str]) -> JobMatch:
    return JobMatch(
        job_id=job_id,
        job_title=f"Job {job_id}",
        score=score,
        semantic=0.6,
        semantic_scaled=0.6,
        skill_portion=0.5,
        matched_skills=matched,
        missing_skills=missing,
    )


MATCHES = MatchResult(
    matches=[
        job(1, 0.85, ["SQL", "Excel"], ["Power BI", "dbt"]),
        job(2, 0.80, ["SQL"], ["Power BI"]),
    ],
    embedding_calls=1,
)


def state(**overrides) -> dict:
    data = {
        "profile": PROFILE,
        "version": 1,
        "matches": MATCHES,
        "file_name": "cv_nadia.pdf",
        "file_size": 120_000,
        "pages": 1,
        "usage": LLMUsage(llm_calls=1),
        "embedding_calls": 1,
    }
    data.update(overrides)
    return data


@pytest.fixture
def plans(monkeypatch):
    """Ganti dependency dan LLM rencana; kembalikan catatan panggilan dan balasan."""
    recorded: dict = {"calls": [], "reply": LearningPlanResult(plan=PLAN, usage=PLAN_USAGE)}

    def fake_generate(profile, gap, *, llm):
        recorded["calls"].append((profile, gap, llm))
        return recorded["reply"]

    monkeypatch.setattr(skill_gap, "load_dependencies", lambda: DEPS)
    monkeypatch.setattr(skill_gap, "generate_learning_plan", fake_generate)
    return recorded


def _page():
    from src.components.skill_gap import render_skill_gap_page

    render_skill_gap_page()


def _app(cv_state: dict | None = None) -> AppTest:
    app = AppTest.from_function(_page, default_timeout=APP_TIMEOUT)
    if cv_state is not None:
        app.session_state[CV_STATE_KEY] = cv_state
    app.run()
    return app


def _bars(app: AppTest) -> list[str]:
    return [element.proto.text for element in app.get("progress")]


def _captions(app: AppTest) -> list[str]:
    return [element.value for element in app.caption]


# --- Tanpa CV ------------------------------------------------------------------


def test_without_cv_points_to_cv_match(plans):
    app = _app()

    assert not app.exception
    assert app.markdown[0].value == PAGE_TITLE
    assert app.info[0].value == NO_CV
    assert app.button(key="gap_to_cv").label == "Buka Cocokkan CV"
    assert _bars(app) == []


def test_failed_match_points_to_cv_match(plans):
    app = _app(state(matches=MatchResult(error="Embedding error: 403", embedding_calls=1)))

    assert app.info[0].value == NO_MATCHES
    assert plans["calls"] == []


# --- Skill gap (0 LLM) -----------------------------------------------------------


def test_bars_show_gap_then_owned_skills_without_llm(plans):
    app = _app(state())

    assert not app.exception
    assert _bars(app) == [
        "Power BI · 2 dari 2",
        "dbt · 1 dari 2",
        "SQL · 2 dari 2",
        "Excel · 1 dari 2",
    ]
    assert [element.proto.value for element in app.get("progress")][:2] == [100, 50]
    assert gap_caption(compute_skill_gap(MATCHES.matches)) in _captions(app)
    assert not app.button(key="gap_plan").disabled
    assert plans["calls"] == []


def test_no_gap_disables_plan_button(plans):
    full = MatchResult(matches=[job(1, 0.9, ["SQL"], [])], embedding_calls=1)
    app = _app(state(matches=full))

    assert NO_GAP in _captions(app)
    assert app.button(key="gap_plan").disabled


# --- Rencana belajar (1 LLM, cache per versi) --------------------------------------


def test_plan_calls_llm_once_and_is_cached_for_the_version(plans):
    app = _app(state())
    app.button(key="gap_plan").click().run()

    assert not app.exception
    assert len(plans["calls"]) == 1
    profile, gap, llm = plans["calls"][0]
    assert (profile, llm) == (PROFILE, "llm")
    assert [skill.name for skill in gap.missing] == ["Power BI", "dbt"]
    assert app.session_state[CV_STATE_KEY][PLAN_STATE_KEY]["version"] == 1
    assert [box.label for box in app.checkbox] == [f"Minggu {n} — Fokus {n}" for n in range(1, 5)]
    assert "- Tugas 1a\n- Tugas 1b" in _captions(app)
    assert app.download_button(key="gap_download").proto.label == "Unduh rencana (.md)"
    assert plan_usage_caption(plans["reply"]) in _captions(app)

    app.checkbox(key="gap_week_1_1").check().run()
    assert len(plans["calls"]) == 1  # rerun tidak memanggil LLM lagi


def test_new_profile_version_needs_new_plan(plans):
    cached = {"version": 1, "result": LearningPlanResult(plan=PLAN, usage=PLAN_USAGE)}
    app = _app(state(version=2, plan=cached))

    assert app.button(key="gap_plan").label == "Buat rencana belajar"
    assert app.checkbox == []
    assert plans["calls"] == []


def test_failed_plan_shows_warning_and_can_retry(plans):
    plans["reply"] = LearningPlanResult(error="LLM error: timeout")
    app = _app(state())
    app.button(key="gap_plan").click().run()

    assert PLAN_FAILED in [element.value for element in app.warning]
    assert "timeout" not in " ".join(_captions(app))
    assert app.button(key="gap_plan").label == "Coba lagi"

    plans["reply"] = LearningPlanResult(plan=PLAN, usage=PLAN_USAGE)
    app.button(key="gap_plan").click().run()
    assert len(plans["calls"]) == 2
    assert len(app.checkbox) == 4


def test_missing_config_shows_error_without_llm(plans, monkeypatch):
    def broken():
        raise KeyError("OPENAI_API_KEY")

    monkeypatch.setattr(skill_gap, "load_dependencies", broken)
    app = _app(state())
    app.button(key="gap_plan").click().run()

    assert not app.exception
    assert app.error[0].value == DEPENDENCY_ERROR
    assert plans["calls"] == []


def test_page_is_registered_in_app(plans):
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    app.run()
    app.switch_page("pages/skillgap_page.py").run()

    assert not app.exception
    assert app.markdown[0].value == PAGE_TITLE
    assert app.info[0].value == NO_CV
    links = [element.proto.label for element in app.sidebar.get("page_link")]
    assert links == ["Beranda", "Cari Lowongan", "Cocokkan CV", "Skill Gap", "Asisten Karier"]


def test_ask_assistant_opens_assistant_with_gap_question(plans, monkeypatch):
    graph = _use(monkeypatch, AgentResult(answer="Power BI: 2 dari 2 lowongan", source="cv"))
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    app.session_state[CV_STATE_KEY] = state()
    app.run()
    app.switch_page("pages/skillgap_page.py").run()
    app.button(key="gap_to_assistant").click().run()

    assert not app.exception
    assert app.markdown[0].value == "## Asisten Karier"
    assert graph.inputs[0]["messages"] == [{"role": "user", "content": ASSISTANT_QUESTION}]
    assert graph.inputs[0]["matches"] == MATCHES
    assert plans["calls"] == []


# --- Helper tanpa Streamlit --------------------------------------------------------


def test_cached_plan_only_for_current_version():
    result = LearningPlanResult(plan=PLAN)

    assert cached_plan(state()) is None
    assert cached_plan(state(plan={"version": 1, "result": result})) is result
    assert cached_plan(state(version=2, plan={"version": 1, "result": result})) is None


def test_labels_escape_markdown():
    assert count_label(SkillCount("C#", 3, 2.0), 10) == "C\\# · 3 dari 10"
    assert week_label(2, PlanWeek(title="SQL *lanjutan*", skills=[], tasks=[])) == (
        "Minggu 2 — SQL \\*lanjutan\\*"
    )


def test_gap_caption_mentions_hidden_skills_and_jobs_without_skills():
    gap = compute_skill_gap(MATCHES.matches, limit=1)
    assert "1 skill lain diminta lebih sedikit lowongan." in gap_caption(gap)

    bare = job(3, 0.5, [], []).model_copy(update={"skill_portion": None})
    assert "1 lowongan tidak menyebut skill spesifik." in gap_caption(compute_skill_gap([bare]))
