"""Halaman surat lamaran dengan Streamlit AppTest (tanpa browser dan tanpa API key).

``generate_cover_letter`` dan ``load_dependencies`` diganti fungsi palsu: yang dites adalah UI,
yaitu syarat CV dan lowongan, satu panggilan LLM per versi profil dan lowongan, "Buat ulang"
dengan pengaturan terpilih, editan yang tersimpan, dan pesan gagal tanpa isi error. Isi prompt
dan format surat dites di test_cv_letter.py.
"""

from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.components import cover_letter, cv_match
from src.components.cover_letter import (
    CONTACT_KEY,
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
    contact_line,
    current_settings,
    find_match,
    letter_caption,
    letter_text,
)
from src.components.cv_match import LETTER_JOB_KEY, CVDependencies
from src.cv.ingest import CV_STATE_KEY
from src.cv.letter import CoverLetter, CoverLetterResult, LetterSettings
from src.cv.match import JobMatch, MatchResult
from src.cv.profile import CVProfile
from src.models.agent import LLMCall, LLMUsage

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"
# Run pertama di satu proses pytest memuat pandas/pyarrow: lebih dari 10 detik di laptop macOS.
APP_TIMEOUT = 30
DEPS = CVDependencies(llm="llm", embeddings="emb", vector_store="store", db_path="db.sqlite")
PROFILE = CVProfile(
    full_name="BUDI CONTOH",
    target_role="Data Analyst",
    location="Malang, Jawa Timur",
    skills=["SQL"],
    contacts=["0812-3456-7890", "budi@mail.com"],
)
LETTER_USAGE = LLMUsage(
    llm_calls=1,
    input_tokens=1200,
    output_tokens=420,
    total_tokens=1620,
    calls=[
        LLMCall(
            step="cv_letter",
            model="gpt-4o-mini-2024-07-18",
            input_tokens=1200,
            output_tokens=420,
            total_tokens=1620,
        )
    ],
)
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
    from src.components.cover_letter import render_cover_letter_page

    render_cover_letter_page()


def _app(cv_state: dict | None = None) -> AppTest:
    app = AppTest.from_function(_page, default_timeout=APP_TIMEOUT)
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
    assert text.startswith("Malang, ")
    assert "Kepada Yth.\nTim Rekrutmen\nPT Alpha\n\n" in text
    assert "Perihal: Lamaran Pekerjaan sebagai Data Analyst\n\nDengan hormat,\n\n" in text
    assert "Dengan hormat,\n\nSaya melamar posisi ini." in text
    assert text.endswith("Hormat saya,\nBudi Contoh")
    assert (
        "Bahasa Indonesia · Standar · 1 panggilan LLM (gpt-4o-mini-2024-07-18) · "
        "1.200 token masuk, 420 keluar · $0.000432."
    ) in _texts(app)
    assert app.selectbox(key=WORDS_KEY).options == ["Singkat", "Standar", "Panjang"]
    assert not app.download_button(key="cv_letter_download").disabled
    assert not app.download_button(key="cv_letter_download_pdf").disabled
    assert app.text_input[0].value == "Malang, Jawa Timur  ·  0812-3456-7890  ·  budi@mail.com"

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
    assert (
        "Re: Application for Data Analyst\n\nDear PT Alpha Hiring Team," in app.text_area[0].value
    )
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
    assert app.download_button(key="cv_letter_download_pdf").disabled
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
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    app.run()
    app.switch_page("pages/cvmatch_page.py")
    app.session_state[CV_STATE_KEY] = state(**{LETTER_JOB_KEY: None})
    app.run()
    app.button(key="cv_letter_3").click().run()

    assert not app.exception
    assert app.session_state[CV_STATE_KEY][LETTER_JOB_KEY] == 3
    assert app.markdown[0].value == "## Graphic Designer"
    assert "Kepada Yth.\nTim Rekrutmen\n\nPerihal: " in app.text_area[0].value


def test_contact_edit_is_kept_for_other_letters(letters):
    app = _app(state())
    app.text_input[0].input("Malang  ·  budi@mail.com").run()

    cv = app.session_state[CV_STATE_KEY]
    assert cv[CONTACT_KEY] == "Malang  ·  budi@mail.com"
    reopened = _app({**cv, LETTER_JOB_KEY: 3})
    assert reopened.text_input[0].value == "Malang  ·  budi@mail.com"


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


def test_letter_text_uses_header_from_job_and_profile_city():
    entry = {
        "settings": LetterSettings(),
        "result": CoverLetterResult(letter=LETTER, usage=LETTER_USAGE),
    }

    text = letter_text(entry, MATCHES.matches[0], PROFILE, today=date(2026, 10, 25))

    assert text.startswith("Malang, 25 Oktober 2026\n\nKepada Yth.\nTim Rekrutmen\nPT Alpha\n\n")


def test_contact_line_skips_empty_parts():
    assert contact_line(PROFILE) == "Malang, Jawa Timur  ·  0812-3456-7890  ·  budi@mail.com"
    assert contact_line(CVProfile(contacts=["a@b.c"])) == "a@b.c"
    assert contact_line(CVProfile()) == ""


def test_letter_caption_shows_settings_and_cost():
    entry = {
        "settings": LetterSettings(language="en", words=150),
        "result": CoverLetterResult(letter=LETTER, usage=LETTER_USAGE),
    }

    assert letter_caption(entry) == (
        "English · Singkat · 1 panggilan LLM (gpt-4o-mini-2024-07-18) · "
        "1.200 token masuk, 420 keluar · $0.000432."
    )
