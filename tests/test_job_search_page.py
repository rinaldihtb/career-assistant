"""Halaman Cari Lowongan dengan Streamlit AppTest, memakai database contoh (6 lowongan).

Logika pencarian sudah dites di test_job_search.py; yang dites di sini adalah form, daftar,
detail, perilaku dengan dan tanpa CV, serta helper format.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.components import job_search
from src.components.job_search import (
    DB_ERROR,
    KEYWORD_KEY,
    LETTER_DISABLED_HELP,
    MODE_KEY,
    NO_RESULTS,
    NOT_IN_MATCHES,
    PAGE_TITLE,
    PROVINCE_KEY,
    SALARY_KEY,
    counted_label,
    detail_badges,
    listing_meta,
    required_skills,
    results_caption,
    salary_option_label,
    short_salary,
)
from src.cv.ingest import CV_STATE_KEY
from src.cv.match import JobMatch, JobSkillIndex, MatchResult
from src.cv.profile import CVProfile
from src.repositories.job_search import JobFilters, JobListing

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"
# Run pertama di satu proses pytest memuat pandas/pyarrow: lebih dari 10 detik di laptop macOS.
APP_TIMEOUT = 30


@pytest.fixture(autouse=True)
def db(monkeypatch, jobs_db, make_repository):
    """Halaman membaca lowongan dari repository fixture dan indeks skill dari ``jobs_db``."""
    repository = make_repository()
    monkeypatch.setattr(job_search, "get_job_repository", lambda: repository)
    monkeypatch.setenv("DB_SQLITE_PATH", str(jobs_db))
    return jobs_db


def cv_state() -> dict:
    match = JobMatch(
        job_id=1,
        job_title="Data Analyst",
        score=0.83,
        semantic=0.7,
        semantic_scaled=0.9,
        skill_portion=1.0,
        matched_skills=["SQL", "Python"],
    )
    return {
        "profile": CVProfile(target_role="Data Analyst", skills=["SQL", "Excel"]),
        "version": 1,
        "matches": MatchResult(matches=[match]),
        "file_name": "cv.pdf",
    }


def _page():
    from src.components.job_search import render_job_search_page

    render_job_search_page()


def _app(cv: dict | None = None) -> AppTest:
    app = AppTest.from_function(_page, default_timeout=APP_TIMEOUT)
    if cv is not None:
        app.session_state[CV_STATE_KEY] = cv
    return app.run()


def _search(app: AppTest) -> AppTest:
    next(button for button in app.button if button.label == "Cari").click()
    return app.run()


def _captions(app: AppTest) -> list[str]:
    return [caption.value for caption in app.caption]


def _markdown(app: AppTest) -> list[str]:
    return [block.value for block in app.markdown]


def _cards(app: AppTest) -> list[str]:
    return [button.key for button in app.button if (button.key or "").startswith("job_pick_")]


# --- Form dan daftar ----------------------------------------------------------


def test_default_lists_all_jobs_and_details_the_first():
    app = _app()

    assert not app.exception
    assert app.markdown[0].value == PAGE_TITLE
    assert "Menampilkan 6 dari 6 lowongan" in _captions(app)
    assert _cards(app) == [f"job_pick_{i}" for i in range(1, 7)]
    assert "### Data Analyst" in _markdown(app)


def test_filter_options_show_counts():
    app = _app()

    assert app.selectbox(key=PROVINCE_KEY).options[:2] == ["Semua", "DKI Jakarta (3)"]
    assert app.selectbox(key=MODE_KEY).options == ["Semua", "Hibrid (1)"]
    assert app.selectbox(key=SALARY_KEY).options[2] == "Minimal Rp10 juta (2)"


def test_keyword_search_runs_on_submit_only():
    app = _app()
    app.text_input(key=KEYWORD_KEY).input("python").run()
    assert "Menampilkan 6 dari 6 lowongan" in _captions(app)

    app.text_input(key=KEYWORD_KEY).input("python")
    _search(app)

    assert "Menampilkan 2 dari 2 lowongan · urut relevansi kata kunci" in _captions(app)
    assert _cards(app) == ["job_pick_1", "job_pick_5"]


def test_filters_combine_and_salary_filter_is_explained():
    app = _app()
    app.selectbox(key=PROVINCE_KEY).set_value("Jakarta Raya")
    app.selectbox(key=SALARY_KEY).set_value(10_000_000)
    _search(app)

    assert _cards(app) == ["job_pick_1", "job_pick_4"]
    assert (
        "Menampilkan 2 dari 2 lowongan · lowongan tanpa info gaji tidak ditampilkan"
        in _captions(app)
    )


def test_no_results_shows_hint():
    app = _app()
    app.text_input(key=KEYWORD_KEY).input("astronot")
    _search(app)

    assert [info.value for info in app.info] == [NO_RESULTS]
    assert _cards(app) == []


def test_clicking_a_card_shows_its_detail():
    app = _app()
    app.button(key="job_pick_3").click().run()

    assert "### Graphic Designer" in _markdown(app)
    assert app.button(key="job_pick_3").icon == ":material/arrow_right:"


def test_show_more_adds_the_next_page(monkeypatch):
    monkeypatch.setattr(job_search, "PAGE_SIZE", 4)
    app = _app()
    assert len(_cards(app)) == 4

    app.button(key="job_search_more").click().run()

    assert len(_cards(app)) == 6
    assert "Menampilkan 6 dari 6 lowongan" in _captions(app)


def test_database_error_is_shown(monkeypatch):
    def broken():
        raise KeyError("QDRANT_URL")

    monkeypatch.setattr(job_search, "get_job_repository", broken)
    app = _app()

    assert [error.value for error in app.error] == [DB_ERROR]


# --- Detail dengan dan tanpa CV -----------------------------------------------


def test_without_cv_shows_required_skills_and_cv_buttons():
    app = _app()

    assert "**Skill yang diminta**" in _markdown(app)
    assert any(
        ":gray-badge[Python]" in text and ":gray-badge[SQL]" in text for text in _markdown(app)
    )
    assert not app.button(key="job_letter_1").disabled
    assert app.button(key="job_fit_1").label == "Cek kecocokan dengan CV"


def test_with_cv_job_in_matches_shows_score_and_enables_letter():
    app = _app(cv_state())

    assert "#### 83%" in _markdown(app)
    assert not app.button(key="job_letter_1").disabled
    assert "**Skill yang diminta**" not in _markdown(app)


def test_with_cv_other_job_shows_word_match_and_disables_letter():
    app = _app(cv_state())
    app.button(key="job_pick_5").click().run()

    assert "Skill cocok: belum ada" in _markdown(app)
    assert "Skill kurang: :orange-badge[Python] :orange-badge[REST API]" in _markdown(app)
    assert NOT_IN_MATCHES in _captions(app)
    letter = app.button(key="job_letter_5")
    assert letter.disabled
    assert letter.help == LETTER_DISABLED_HELP


def test_page_is_registered_and_fit_button_opens_cv_match():
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    app.run()
    app.switch_page("pages/jobsearch_page.py").run()

    assert not app.exception
    assert app.markdown[0].value == PAGE_TITLE

    app.button(key="job_fit_1").click().run()
    assert app.markdown[0].value == "## Cocokkan CV dengan lowongan"


# --- Helper tanpa Streamlit ---------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (8_000_000, 10_000_000, "Rp8–10 jt"),
        (7_500_000, 7_500_000, "Rp7,5 jt"),
        (None, 12_000_000, "Rp12 jt"),
        (None, None, "Gaji tidak dicantumkan"),
    ],
)
def test_short_salary(low, high, expected):
    assert short_salary(low, high) == expected


def _listing(**overrides) -> JobListing:
    data = {
        "id": 4,
        "job_title": "Marketing Manager",
        "company_name": "PT Delta",
        "location": "Jakarta Raya\n(Hibrid)",
        "work_type": "Full time",
        "salary_min": 15_000_000,
        "salary_max": 20_000_000,
        "province": "Jakarta Raya",
        "mode": "Hibrid",
    }
    data.update(overrides)
    return JobListing(**data)


def test_listing_meta_and_badges():
    job = _listing()

    assert listing_meta(job) == "DKI Jakarta · Full time · Hibrid · Rp15–20 jt"
    assert detail_badges(job) == (
        ":gray-badge[Full time] :blue-badge[Hibrid] :green-badge[Rp15–20 jt] :gray-badge[ID 4]"
    )
    bare = _listing(work_type=None, mode=None, salary_min=None, salary_max=None)
    assert detail_badges(bare) == ":orange-badge[Gaji tidak dicantumkan] :gray-badge[ID 4]"


def test_required_skills_most_common_first():
    index = JobSkillIndex(
        vocab={1: frozenset({"Tableau", "SQL", "Excel"})},
        descriptions={},
        frequency={"SQL": 50, "Excel": 80, "Tableau": 10},
    )

    assert required_skills(1, index) == ["Excel", "SQL", "Tableau"]
    assert required_skills(2, index) == []
    assert required_skills(1, None) == []


def test_results_caption_and_option_labels():
    assert results_caption(10, 73, JobFilters()) == "Menampilkan 10 dari 73 lowongan"
    assert salary_option_label(0, 473) == "Semua"
    assert salary_option_label(5_000_000, 152) == "Minimal Rp5 juta (152)"
    label = counted_label([("Jakarta Raya", 316)], lambda name: f"[{name}]")
    assert label("") == "Semua"
    assert label("Jakarta Raya") == "[Jakarta Raya] (316)"
