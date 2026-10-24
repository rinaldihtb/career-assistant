"""Halaman surat lamaran dengan Streamlit AppTest (tanpa browser dan tanpa API key).

``generate_cover_letter`` dan ``load_dependencies`` diganti fungsi palsu: yang dites adalah UI,
yaitu syarat CV dan lowongan, satu panggilan LLM per versi profil dan lowongan, "Buat ulang"
dengan pengaturan terpilih, editan yang tersimpan, dan pesan gagal tanpa isi error. Isi prompt
dan format surat dites di test_cv_letter.py.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from components import cover_letter, cv_match
from components.cover_letter import (
    DEPENDENCY_ERROR,
    JOB_GONE,
    LANGUAGE_KEY,
    LETTER_FAILED,
    LETTERS_KEY,
    NAME_MISSING,
    NO_CV,
    NO_JOB,
    WORDS_KEY,
    cached_letter,
    current_settings,
    find_match,
    letter_caption,
)
from components.cv_match import LETTER_JOB_KEY, CVDependencies
from cv.ingest import CV_STATE_KEY
from cv.letter import CoverLetter, CoverLetterResult, LetterSettings
from cv.match import JobMatch, MatchResult
from cv.profile import CVProfile
from models.agent import LLMUsage

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"
DEPS = CVDependencies(llm="llm", embeddings="emb", vector_store="store", db_path="db.sqlite")
PROFILE = CVProfile(full_name="BUDI CONTOH", target_role="Data Analyst", skills=["SQL"])
LETTER_USAGE = LLMUsage(llm_calls=1, input_tokens=1200, output_tokens=420, total_tokens=1620)
LETTER = CoverLetter(paragraphs=["Saya melamar posisi ini.", "Saya terbiasa memakai SQL."])


def job(job_id: int, score: float, title: str, company: str | None) -> JobMatch:
    return JobMatch(
        job_id=job_id,
        job_title=title,
        company_name=company,
        score=score,
        semantic=0.6,
        semantic_scaled=0.6,
        skill_portion=0.5,
        matched_skills=["SQL"],
        missing_skills=["Power BI"],
    )


MATCHES = MatchResult(
    matches=[job(1, 0.83, "Data Analyst", "PT Alpha"), job(3, 0.61, "Graphic Designer", None)],
    embedding_calls=1,
)


def state(**overrides) -> dict:
    data = {
        "profile": PROFILE,
        "version": 1,
        "matches": MATCHES,
        "file_name": "cv_budi.pdf",
        "file_size": 186_000,
        "pages": 1,
        "usage": LLMUsage(llm_calls=1),
        "embedding_calls": 1,
        LETTER_JOB_KEY: 1,
    }
    data.update(overrides)
    return data


@pytest.fixture
def letters(monkeypatch, jobs_db):
    """Ganti dependency dan LLM surat; kembalikan catatan panggilan dan balasan."""
    recorded: dict = {
        "calls": [],
        "reply": CoverLetterResult(letter=LETTER, usage=LETTER_USAGE),
    }

    def fake_generate(profile, match, description, *, llm, settings=None):
        recorded["calls"].append(
            {"profile": profile, "match": match, "description": description, "settings": settings}
        )
        return recorded["reply"]

    monkeypatch.setattr(cover_letter, "load_dependencies", lambda: DEPS)
    monkeypatch.setattr(cover_letter, "generate_cover_letter", fake_generate)
    monkeypatch.setattr(cv_match, "load_dependencies", lambda: DEPS)
    monkeypatch.setenv("DB_SQLITE_PATH", str(jobs_db))
    return recorded


def _page():
    from components.cover_letter import render_cover_letter_page

    render_cover_letter_page()


def _app(cv_state: dict | None = None) -> AppTest:
    app = AppTest.from_function(_page, default_timeout=10)
    if cv_state is not None:
        app.session_state[CV_STATE_KEY] = cv_state
    app.run()
    return app


def _texts(app: AppTest) -> list[str]:
    elements = [*app.markdown, *app.caption, *app.info, *app.warning, *app.error]
    return [element.value for element in elements]


# --- Tanpa CV atau tanpa lowongan ----------------------------------------------


@pytest.mark.parametrize(
    ("cv_state", "message"),
    [
        (None, NO_CV),
        (state(**{LETTER_JOB_KEY: None}), NO_JOB),
        (state(**{LETTER_JOB_KEY: 99}), JOB_GONE),
    ],
)
def test_missing_cv_or_job_points_to_cv_match(letters, cv_state, message):
    app = _app(cv_state)

    assert not app.exception
    assert app.info[0].value == message
    assert app.button(key="letter_to_cv").label == "Buka Cocokkan CV"
    assert letters["calls"] == []


# --- Surat --------------------------------------------------------------------


def test_opening_writes_letter_once_with_default_settings(letters):
    app = _app(state())

    assert not app.exception
    assert app.markdown[0].value == "## Data Analyst"
    assert [tab.label for tab in app.tabs] == ["Ringkasan", "Kecocokan CV · 83%", "Surat lamaran"]
    assert len(letters["calls"]) == 1
    call = letters["calls"][0]
    assert call["match"].job_id == 1
    assert call["description"] == "Analisis data penjualan dengan SQL dan Python."
    assert call["settings"] == LetterSettings()
    text = app.text_area[0].value
    assert text.startswith("Yth. Tim Rekrutmen PT Alpha,\n\nSaya melamar posisi ini.")
    assert text.endswith("Hormat saya,\nBudi Contoh")
    assert "Bahasa Indonesia · ± 250 kata · 1 panggilan LLM (1.620 token)." in _texts(app)
    assert not app.download_button(key="cv_letter_download").disabled

    app.run()
    assert len(letters["calls"]) == 1


def test_regenerate_uses_selected_language_and_length(letters):
    app = _app(state())
    app.selectbox(key=LANGUAGE_KEY).set_value("en")
    app.selectbox(key=WORDS_KEY).set_value(150)
    app.button(key="cv_letter_regenerate").click().run()

    assert len(letters["calls"]) == 2
    assert letters["calls"][1]["settings"] == LetterSettings(language="en", words=150)
    entry = app.session_state[CV_STATE_KEY][LETTERS_KEY][1]
    assert entry["round"] == 2
    assert app.text_area[0].value.startswith("Dear PT Alpha Hiring Team,")
    assert app.text_area[0].value.endswith("Sincerely,\nBudi Contoh")


def test_edits_are_kept_after_leaving_the_page(letters):
    app = _app(state())
    app.text_area[0].input("Surat versi saya.").run()

    cv = app.session_state[CV_STATE_KEY]
    assert cv[LETTERS_KEY][1]["text"] == "Surat versi saya."
    assert "3 kata · bisa diedit sebelum diunduh." in _texts(app)

    reopened = _app(cv)
    assert reopened.text_area[0].value == "Surat versi saya."
    assert len(letters["calls"]) == 1


def test_profile_change_writes_a_new_letter(letters):
    app = _app(state())
    cv = app.session_state[CV_STATE_KEY]

    _app({**cv, "version": 2})

    assert len(letters["calls"]) == 2


def test_failure_shows_message_without_error_text(letters):
    letters["reply"] = CoverLetterResult(usage=LETTER_USAGE, error="Output LLM tidak valid: BUDI")
    app = _app(state())

    assert app.warning[0].value == LETTER_FAILED
    assert not any("BUDI" in text for text in _texts(app))
    assert app.download_button(key="cv_letter_download").disabled
    app.run()
    assert len(letters["calls"]) == 1


def test_dependency_error_skips_llm(letters, monkeypatch):
    def broken():
        raise KeyError("OPENAI_API_KEY")

    monkeypatch.setattr(cover_letter, "load_dependencies", broken)
    app = _app(state())

    assert not app.exception
    assert app.error[0].value == DEPENDENCY_ERROR
    assert letters["calls"] == []


def test_missing_name_uses_placeholder_and_warns(letters):
    app = _app(state(profile=PROFILE.model_copy(update={"full_name": None})))

    assert app.text_area[0].value.endswith("Hormat saya,\n[Nama Anda]")
    assert NAME_MISSING in _texts(app)


def test_card_button_opens_registered_letter_page(letters):
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.run()
    app.switch_page("pages/cvmatch.page.py")
    app.session_state[CV_STATE_KEY] = state(**{LETTER_JOB_KEY: None})
    app.run()
    app.button(key="cv_letter_3").click().run()

    assert not app.exception
    assert app.session_state[CV_STATE_KEY][LETTER_JOB_KEY] == 3
    assert app.markdown[0].value == "## Graphic Designer"
    assert app.text_area[0].value.startswith("Yth. Tim Rekrutmen,\n\n")


# --- Helper tanpa Streamlit ----------------------------------------------------


def test_find_match_and_cached_letter_follow_current_version():
    entry = {"version": 1, "settings": LetterSettings(), "result": None, "round": 1}
    cv = state(**{LETTERS_KEY: {1: entry}})

    assert find_match(cv, 3).job_title == "Graphic Designer"
    assert find_match(cv, 99) is None
    assert cached_letter(cv, 1) is entry
    assert cached_letter({**cv, "version": 2}, 1) is None
    assert cached_letter(state(), 1) is None


def test_current_settings_default_and_selected():
    assert current_settings({}) == LetterSettings()
    assert current_settings({LANGUAGE_KEY: "en", WORDS_KEY: 350}) == LetterSettings(
        language="en", words=350
    )


def test_letter_caption_shows_settings_and_cost():
    entry = {
        "settings": LetterSettings(language="en", words=150),
        "result": CoverLetterResult(letter=LETTER, usage=LETTER_USAGE),
    }

    assert letter_caption(entry) == "English · ± 150 kata · 1 panggilan LLM (1.620 token)."
