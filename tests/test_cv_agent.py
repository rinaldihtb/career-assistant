"""Agent CV: jawaban dari hasil Cocokkan CV tanpa LLM (agents/cv_agent.py)."""

from agents import cv_agent
from agents.cv_agent import CV_MATCH_LIMIT, profile_label, run_cv_agent
from agents.prompts import (
    CV_FAILED_MESSAGE,
    CV_GAP_PLAN,
    CV_NO_MATCHES_MESSAGE,
)
from components.assistant_chat import split_rag_answer
from cv.match import JobMatch, MatchResult
from cv.profile import CVProfile

PROFILE = CVProfile(
    full_name="Nadia Putri",
    target_role="Data Analyst",
    location="Bandung",
    skills=["SQL", "Excel"],
    experience_months=14,
)


def job(job_id: int, score: float, matched=(), missing=(), **fields) -> JobMatch:
    data = {
        "job_id": job_id,
        "job_title": f"Data Analyst {job_id}",
        "company_name": f"PT {job_id}",
        "location": "Jakarta Selatan, Jakarta Raya",
        "score": score,
        "semantic": 0.6,
        "semantic_scaled": 0.6,
        "skill_portion": 0.5,
        "matched_skills": list(matched),
        "missing_skills": list(missing),
    }
    data.update(fields)
    return JobMatch(**data)


def matches(*jobs: JobMatch) -> MatchResult:
    return MatchResult(matches=list(jobs), embedding_calls=1)


# --- cv_match --------------------------------------------------------------------


def test_match_lists_top_jobs_in_rag_line_format():
    result = run_cv_agent(
        "Lowongan yang paling cocok dengan CV-ku",
        kind="cv_match",
        profile=PROFILE,
        matches=matches(
            job(8, 0.82, matched=["SQL", "Excel", "Python", "Tableau"]),
            job(28, 0.776, location="Jakarta Raya\n(Hibrid)", company_name=None),
        ),
    )

    assert result.source == "cv"
    assert result.job_ids == [8, 28]
    assert result.usage.llm_calls == 0
    assert result.error is None
    intro = (
        "Dari profilmu (Data Analyst · 1-3 tahun), ini 2 teratas dari 2 lowongan hasil Cocokkan "
        "CV (daftar lengkap ada di halaman Cocokkan CV). Skor = 60% kemiripan profil dengan "
        "deskripsi lowongan + 40% porsi skill yang cocok:"
    )
    first = (
        "- [ID:8] Data Analyst 8 - PT 8 (Jakarta Selatan, Jakarta Raya): 82% cocok · "
        "skill cocok: SQL, Excel, Python"
    )
    second = "- [ID:28] Data Analyst 28 - - (Jakarta Raya (Hibrid)): 78% cocok"
    assert result.answer.splitlines() == [intro, first, second]


def test_match_lines_are_parsed_by_chat_cards():
    result = run_cv_agent(
        "x", kind="cv_match", profile=PROFILE, matches=matches(job(8, 0.82, matched=["SQL"]))
    )
    intro, notes = split_rag_answer(result.answer)

    assert intro.startswith("Dari profilmu")
    assert notes == {8: "82% cocok · skill cocok: SQL"}


def test_match_shows_limit_and_points_to_full_list():
    jobs = [job(job_id, 0.9 - job_id / 100) for job_id in range(1, 11)]
    result = run_cv_agent("x", kind="cv_match", profile=PROFILE, matches=matches(*jobs))
    lines = result.answer.splitlines()

    assert result.job_ids == list(range(1, CV_MATCH_LIMIT + 1))
    assert "ini 5 teratas dari 10 lowongan hasil Cocokkan CV (daftar lengkap" in lines[0]
    assert len(lines) == 1 + CV_MATCH_LIMIT
    assert lines[-1].startswith("- [ID:5]")  # baris terakhir tetap baris lowongan


def test_match_without_profile_label():
    result = run_cv_agent("x", kind="cv_match", profile=None, matches=matches(job(1, 0.5)))

    assert result.answer.startswith("Dari profilmu, ini 1 teratas dari 1 lowongan")


# --- cv_gap ----------------------------------------------------------------------


def test_gap_counts_missing_and_owned_skills():
    result = run_cv_agent(
        "Skill yang perlu kupelajari dari CV-ku",
        kind="cv_gap",
        profile=PROFILE,
        matches=matches(
            job(1, 0.85, matched=["SQL", "Excel"], missing=["Power BI", "dbt"]),
            job(2, 0.80, matched=["SQL"], missing=["Power BI"]),
        ),
    )

    assert result.source == "cv"
    assert result.job_ids == []
    assert result.answer == (
        "Dari 2 lowongan yang paling cocok dengan CV-mu, skill yang paling sering diminta "
        "dan belum ada di CV:\n"
        "- Power BI: 2 dari 2 lowongan\n"
        "- dbt: 1 dari 2 lowongan\n\n"
        "Skill CV-mu yang paling sering diminta: SQL (2 dari 2), Excel (1 dari 2).\n\n"
        f"{CV_GAP_PLAN}"
    )


def test_gap_without_missing_skills():
    result = run_cv_agent(
        "x", kind="cv_gap", profile=PROFILE, matches=matches(job(1, 0.9, matched=["SQL"]))
    )

    assert result.answer == (
        "Semua skill yang disebut 1 lowongan teratasmu sudah ada di CV-mu.\n\n"
        "Skill CV-mu yang paling sering diminta: SQL (1 dari 1)."
    )


# --- Tanpa hasil / gagal -----------------------------------------------------------


def test_missing_or_failed_matches_point_to_cv_page():
    for result in (None, MatchResult(error="Embedding error: 403"), MatchResult()):
        answer = run_cv_agent("x", kind="cv_match", profile=PROFILE, matches=result)
        assert answer.answer == CV_NO_MATCHES_MESSAGE
        assert answer.source == "cv"
        assert answer.job_ids == []
        assert answer.error is None


def test_unexpected_error_is_returned_not_raised(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("rusak")

    monkeypatch.setattr(cv_agent, "compute_skill_gap", broken)
    result = run_cv_agent("x", kind="cv_gap", profile=PROFILE, matches=matches(job(1, 0.9)))

    assert result.answer == CV_FAILED_MESSAGE
    assert result.error == "CV agent error: rusak"


def test_profile_label():
    assert profile_label(PROFILE) == "Data Analyst · 1-3 tahun"
    assert profile_label(CVProfile(target_role="  Data\nAnalyst ")) == "Data Analyst"
    assert profile_label(CVProfile(experience_months=0)) == "0-1 tahun"
    assert profile_label(CVProfile()) == ""
    assert profile_label(None) == ""
