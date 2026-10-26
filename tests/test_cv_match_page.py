"""Halaman Cocokkan CV dengan Streamlit AppTest (tanpa browser dan tanpa API key).

``ingest_cv`` dan ``match_jobs`` diganti fungsi palsu: yang dites adalah UI, yaitu syarat
persetujuan, satu kali baca dan cocokkan, koreksi profil, versi, dan "Ganti". Logika ekstraksi
dan pencocokan sudah dites di test_cv_ingest.py dan test_cv_match.py.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.components import cv_match
from src.components.cv_match import (
    CONSENT_KEY,
    DEPENDENCY_ERROR,
    EMPTY_RESULTS,
    MATCH_FAILED,
    PAGE_TITLE,
    REVIEW_HINT,
    CVDependencies,
    edit_profile,
    match_meta,
    needs_rematch,
    score_formula,
    size_label,
    skill_chips,
    skill_lines,
    usage_caption,
)
from src.cv.ingest import CV_STATE_KEY, IngestResult
from src.cv.match import JobMatch, MatchConfig, MatchResult
from src.cv.profile import CVProfile, Experience
from src.models.agent import LLMUsage

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"
# Run pertama di satu proses pytest memuat pandas/pyarrow untuk st.dataframe: 7 detik di
# workspace dan lebih dari 10 detik di laptop macOS. Run berikutnya di bawah 1 detik.
APP_TIMEOUT = 30
CV_PDF = Path("tests/eval/cv/cv01_data_analyst.pdf")
DEPS = CVDependencies(llm="llm", embeddings="emb", vector_store="store", db_path="db.sqlite")
PROFILE = CVProfile(
    full_name="BUDI CONTOH",
    target_role="Data Analyst",
    location="Surabaya",
    education_level="S1",
    education_major="Statistika",
    skills=["SQL", "Python", "Power BI"],
    experience_months=30,
    experiences=[Experience(title="Data Analyst", company="PT A", start="2024-01", end="present")],
)
USAGE = LLMUsage(llm_calls=1, input_tokens=1000, output_tokens=200, total_tokens=1200)


def job(job_id: int, score: float, **overrides) -> JobMatch:
    data = {
        "job_id": job_id,
        "job_title": "Data Analyst",
        "company_name": "PT Alpha",
        "location": "Jakarta Selatan, Jakarta Raya\n(Hibrid)",
        "work_type": "Full time",
        "salary_min": 8_000_000,
        "salary_max": 10_000_000,
        "score": score,
        "semantic": 0.66,
        "semantic_scaled": 0.84,
        "skill_portion": 0.8,
        "skill_score": 0.57,
        "matched_skills": ["SQL", "Python", "Excel", "Pandas"],
        "missing_skills": ["Power BI"],
    }
    data.update(overrides)
    return JobMatch(**data)


MATCHES = MatchResult(
    matches=[
        job(1, 0.83),
        job(3, 0.61, job_title="Graphic Designer", skill_portion=None, matched_skills=[],
            missing_skills=[], salary_min=None, salary_max=None),
    ],
    embedding_calls=1,
)  # fmt: skip


def state(**overrides) -> dict:
    data = {
        "profile": PROFILE,
        "version": 1,
        "matches": MATCHES,
        "file_name": "cv_budi.pdf",
        "file_size": 186_000,
        "pages": 1,
        "usage": USAGE,
        "embedding_calls": 1,
    }
    data.update(overrides)
    return data


@pytest.fixture
def calls(monkeypatch, jobs_db):
    """Ganti dependency, ingest, dan match; kembalikan catatan setiap panggilan."""
    recorded: dict[str, list] = {"ingest": [], "match": []}
    replies = {"ingest": IngestResult(profile=PROFILE, usage=USAGE, pages=1, chars=1500)}

    def fake_ingest(data, *, llm, today=None):
        recorded["ingest"].append(data)
        return replies["ingest"]

    def fake_match(profile, **kwargs):
        recorded["match"].append(profile)
        return MATCHES

    monkeypatch.setattr(cv_match, "load_dependencies", lambda: DEPS)
    monkeypatch.setattr(cv_match, "ingest_cv", fake_ingest)
    monkeypatch.setattr(cv_match, "match_jobs", fake_match)
    monkeypatch.setenv("DB_SQLITE_PATH", str(jobs_db))
    recorded["replies"] = replies
    return recorded


def _page():
    from src.components.cv_match import render_cv_match_page

    render_cv_match_page()


def _app(cv_state: dict | None = None) -> AppTest:
    app = AppTest.from_function(_page, default_timeout=APP_TIMEOUT)
    if cv_state is not None:
        app.session_state[CV_STATE_KEY] = cv_state
    app.run()
    return app


def _upload(app: AppTest, consent: bool = True) -> AppTest:
    app.file_uploader[0].set_value((CV_PDF.name, CV_PDF.read_bytes(), "application/pdf"))
    app.run()
    if consent:
        app.checkbox(key=CONSENT_KEY).check().run()
    return app


def _markdown(app: AppTest) -> list[str]:
    return [element.value for element in app.markdown]


# --- Upload -------------------------------------------------------------------


def test_empty_page_waits_for_file_and_consent(calls):
    app = _app()

    assert not app.exception
    assert app.markdown[0].value == PAGE_TITLE
    assert app.button(key="cv_read").disabled
    assert app.info[0].value == EMPTY_RESULTS
    assert app.caption[1].value == score_formula()


def test_file_without_consent_cannot_be_read(calls):
    app = _upload(_app(), consent=False)

    assert app.button(key="cv_read").disabled
    assert calls["ingest"] == []


def test_read_cv_calls_llm_once_and_match_once(calls):
    app = _upload(_app())
    assert not app.button(key="cv_read").disabled

    app.button(key="cv_read").click().run()

    assert not app.exception
    assert calls["ingest"] == [CV_PDF.read_bytes()]
    assert calls["match"] == [PROFILE]
    cv = app.session_state[CV_STATE_KEY]
    assert (cv["version"], cv["file_name"], cv["embedding_calls"]) == (1, CV_PDF.name, 1)
    assert "**Data Analyst**" in _markdown(app)
    assert "### 83%" in _markdown(app)
    assert app.warning[0].value == REVIEW_HINT


def test_rejected_cv_shows_message_and_stays_on_upload(calls):
    calls["replies"]["ingest"] = IngestResult(error="File ini bukan PDF.", reason="not_pdf")
    app = _upload(_app())
    app.button(key="cv_read").click().run()

    assert app.error[0].value == "File ini bukan PDF."
    assert CV_STATE_KEY not in app.session_state
    assert calls["match"] == []


def test_missing_config_shows_error_on_read(calls, monkeypatch):
    def broken():
        raise KeyError("OPENAI_API_KEY")

    monkeypatch.setattr(cv_match, "load_dependencies", broken)
    app = _upload(_app())
    app.button(key="cv_read").click().run()

    assert not app.exception
    assert app.error[0].value == DEPENDENCY_ERROR
    assert calls["ingest"] == []


# --- Hasil dan koreksi -------------------------------------------------------


def test_result_cards_show_score_skills_detail_and_letter_button(calls):
    app = _app(state())

    assert not app.exception
    markdown = _markdown(app)
    assert (
        "Skill cocok: :green-badge[SQL] :green-badge[Python] :green-badge[Excel] :gray-badge[+1]"
    ) in markdown
    assert "Skill kurang: :orange-badge[Power BI]" in markdown
    assert any("tidak menyebut skill spesifik" in line for line in markdown)
    assert not app.button(key="cv_letter_1").disabled
    assert app.expander[0].label == "Lihat detail lowongan"
    assert any("Analisis data penjualan" in line for line in markdown)  # deskripsi job 1
    assert usage_caption(state()) in [caption.value for caption in app.caption]


def test_target_change_bumps_version_and_rematches_once(calls):
    app = _app(state())
    app.text_input(key="cv_target_1").input("Data Scientist")
    app.button[1].click().run()  # [0] Ganti, [1] Cocokkan ulang (form)

    cv = app.session_state[CV_STATE_KEY]
    assert cv["version"] == 2
    assert cv["profile"].target_role == "Data Scientist"
    assert cv["embedding_calls"] == 2
    assert [profile.target_role for profile in calls["match"]] == ["Data Scientist"]


def test_location_only_change_bumps_version_without_embedding(calls):
    app = _app(state())
    app.text_input(key="cv_location_1").input("  Jakarta  ")
    app.button[1].click().run()

    cv = app.session_state[CV_STATE_KEY]
    assert (cv["version"], cv["profile"].location, cv["embedding_calls"]) == (2, "Jakarta", 1)
    assert calls["match"] == []


def test_unchanged_profile_does_not_rematch(calls):
    app = _app(state())
    app.button[1].click().run()

    assert app.session_state[CV_STATE_KEY]["version"] == 1
    assert calls["match"] == []


def test_failed_match_shows_warning_and_retries_without_changes(calls):
    app = _app(state(matches=MatchResult(error="Embedding error: 403", embedding_calls=1)))
    assert MATCH_FAILED in [element.value for element in app.warning]

    app.button[1].click().run()

    assert len(calls["match"]) == 1
    assert app.session_state[CV_STATE_KEY]["version"] == 1  # profil sama, hanya coba lagi


def test_replace_clears_cv_and_asks_consent_again(calls):
    app = _app(state())
    app.session_state[CONSENT_KEY] = True
    app.button(key="cv_replace").click().run()

    assert CV_STATE_KEY not in app.session_state
    assert app.session_state[CONSENT_KEY] is False
    assert app.button(key="cv_read").disabled
    assert len(app.file_uploader) == 1


def test_page_is_registered_and_sidebar_shows_cv_status(calls):
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    app.run()
    app.switch_page("pages/cvmatch_page.py").run()

    assert not app.exception
    assert app.markdown[0].value == PAGE_TITLE
    assert [caption.value for caption in app.sidebar.caption] == ["Belum ada CV"]

    app.session_state[CV_STATE_KEY] = state()
    app.run()
    assert [caption.value for caption in app.sidebar.caption] == ["CV aktif: cv_budi.pdf"]


# --- Helper tanpa Streamlit --------------------------------------------------


def test_edit_profile_cleans_input_and_keeps_other_fields():
    edited = edit_profile(
        PROFILE,
        target_role="  Data   Scientist ",
        education_level=None,
        education_major="",
        location=" Bandung ",
        skills=["SQL", "sql", " Tableau ", ""],
    )
    assert edited.target_role == "Data Scientist"
    assert edited.education_level is None
    assert edited.education_major is None
    assert edited.location == "Bandung"
    assert edited.skills == ["SQL", "Tableau"]
    assert (edited.full_name, edited.experiences) == (PROFILE.full_name, PROFILE.experiences)


def test_needs_rematch_only_when_embedded_text_changes():
    assert not needs_rematch(PROFILE, PROFILE.model_copy(update={"location": "Bandung"}))
    assert needs_rematch(PROFILE, PROFILE.model_copy(update={"skills": ["SQL"]}))
    assert needs_rematch(PROFILE, PROFILE.model_copy(update={"education_level": "S2"}))


def test_score_formula_follows_config_weight():
    assert score_formula(MatchConfig(semantic_weight=0.6)).startswith("Skor = 60% kemiripan")
    assert "40% porsi skill" in score_formula(MatchConfig(semantic_weight=0.6))
    assert "tanpa LLM" in score_formula()


def test_match_meta_skips_empty_parts():
    assert match_meta(job(1, 0.8)) == (
        "PT Alpha · Jakarta Selatan, Jakarta Raya (Hibrid) · Full time · "
        "Rp8.000.000 - Rp10.000.000 per bulan"
    )
    bare = job(2, 0.5, company_name=None, location=None, salary_min=None, salary_max=None)
    assert match_meta(bare) == "Full time"


def test_skill_chips_limit_and_escape():
    assert skill_chips(["A", "B", "C", "D", "E"], "green") == (
        ":green-badge[A] :green-badge[B] :green-badge[C] :gray-badge[+2]"
    )
    assert skill_chips(["Excel [VBA]", "C$"], "orange") == (
        r":orange-badge[Excel (VBA)] :orange-badge[C\$]"
    )
    assert skill_chips([], "green") == ""


def test_skill_lines_without_skills_in_job():
    lines = skill_lines(job(1, 0.6, matched_skills=[], missing_skills=["SQL"]))
    assert lines == ["Skill cocok: belum ada", "Skill kurang: :orange-badge[SQL]"]


def test_size_label():
    assert size_label(186_000) == "182 KB"
    assert size_label(300) == "1 KB"
    assert size_label(1_300_000) == "1,2 MB"
