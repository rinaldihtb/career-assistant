"""Runner eval Asisten ber-CV (src/evaluation/run_cv_chat_eval.py) tanpa LLM dan tanpa Qdrant."""

import json
from pathlib import Path

from src.agents.supervisor_agent_graph import AGENT_NODES
from src.cv.match import JobMatch, MatchResult
from src.cv.profile import CVProfile
from src.evaluation.run_cv_chat_eval import (
    CASES_PATH,
    expected_cv,
    load_cases,
    load_profiles,
    rebuild_matches,
    run_case,
    score,
    summarize,
)
from src.models.agent import AgentResult, LLMCall, LLMUsage

PROFILE = CVProfile(target_role="Data Analyst", skills=["SQL", "Python"])


def job(job_id: int, missing=()) -> JobMatch:
    return JobMatch(
        job_id=job_id,
        job_title=f"Job {job_id}",
        score=0.8,
        semantic=0.6,
        semantic_scaled=0.6,
        missing_skills=list(missing),
    )


MATCHES = MatchResult(matches=[job(i, ["Power BI"] if i < 3 else ["dbt"]) for i in range(1, 8)])


def run(**overrides) -> dict:
    data = {
        "response": "answer",
        "agents": ["cv_match"],
        "standalone": ["Lowongan yang paling cocok dengan CV-ku"],
        "answer": "- [ID:1] Job 1\n- [ID:2] Job 2",
        "cv_ids": [1, 2, 3, 4, 5],
        "valid_ids": [1, 2, 3, 4, 5],
        "llm_calls": 1,
        "planner_tokens": 3500,
        "total_tokens": 3500,
        "seconds": 1.2,
        "error": None,
        "sql": [],
    }
    data.update(overrides)
    return data


def case(**overrides) -> dict:
    data = {
        "id": "C1-01",
        "category": "C1",
        "cv": "cv01",
        "question": "Lowongan yang cocok untukku?",
        "expected": {"agents": ["cv_match"]},
    }
    data.update(overrides)
    return data


def _failed(record: dict) -> list[str]:
    return [check["name"] for check in record["checks"] if not check["passed"]]


# --- Hasil Cocokkan CV dibangun ulang tanpa embedding -----------------------------


def test_rebuild_matches_ranks_stored_candidates_with_rank_jobs(jobs_db):
    candidates = [
        {"job_id": 3, "max": 0.55},
        {"job_id": 1, "max": 0.68},
        {"job_id": 999, "max": 0.70},  # vektor basi: tidak ada di SQLite
    ]
    result = rebuild_matches(PROFILE, candidates, jobs_db)

    assert [match.job_id for match in result.matches] == [1, 3]
    assert result.matches[0].job_title == "Data Analyst"
    assert result.matches[0].semantic == 0.68
    assert set(result.matches[0].matched_skills) == {"SQL", "Python"}
    assert result.unique_jobs == 3


def test_expected_cv_is_top_five_and_top_gap():
    assert expected_cv(MATCHES) == {"ids": [1, 2, 3, 4, 5], "gap": ["dbt", "Power BI"]}
    assert expected_cv(None) == {"ids": [], "gap": []}


# --- Penilaian -------------------------------------------------------------------


def test_cv_match_passes_with_top_five_ids():
    record = score(case(), run(), expected_cv(MATCHES))

    assert record["passed"] is True
    assert record["route_ok"] is True
    assert record["llm_budget"] == 1


def test_cv_match_with_other_ids_fails():
    record = score(case(), run(cv_ids=[2, 1, 3, 4, 5]), expected_cv(MATCHES))

    assert record["passed"] is False
    assert _failed(record) == ["cv_match"]


def test_cv_gap_needs_top_skills_in_answer():
    gap_case = case(expected={"agents": ["cv_gap"]})
    good = run(agents=["cv_gap"], answer="- dbt: 5 dari 7\n- Power BI: 2 dari 7", cv_ids=[])
    bad = run(agents=["cv_gap"], answer="- dbt: 5 dari 7", cv_ids=[])

    assert score(gap_case, good, expected_cv(MATCHES))["passed"] is True
    assert _failed(score(gap_case, bad, expected_cv(MATCHES))) == ["cv_gap"]


def test_wrong_response_route_and_invalid_citation_fail():
    record = score(
        case(expected={"response": "refuse", "agents": []}),
        run(answer="- [ID:77] Job", valid_ids=[1]),
        expected_cv(MATCHES),
    )

    assert record["route_ok"] is False
    assert set(_failed(record)) == {"response", "citations"}


def test_standalone_all_and_none_are_checked():
    personalized = case(
        expected={"agents": ["rag"]},
        checks={"standalone_all": [["data analyst"], ["jakarta"]]},
    )
    ok = run(agents=["rag"], standalone=["Lowongan data analyst di Jakarta"], cv_ids=[])
    missing_city = run(agents=["rag"], standalone=["Lowongan data analyst"], cv_ids=[])
    plain = case(expected={"agents": ["sql"]}, checks={"standalone_none": ["data analyst"]})
    leaked = run(agents=["sql"], standalone=["Berapa lowongan data analyst?"], cv_ids=[])

    assert score(personalized, ok, expected_cv(MATCHES))["passed"] is True
    assert _failed(score(personalized, missing_city, expected_cv(MATCHES))) == ["standalone_all"]
    assert _failed(score(plain, leaked, expected_cv(MATCHES))) == ["standalone_none"]


def test_report_only_case_is_not_scored():
    record = score(
        case(cv=None, expected={"report_only": True, "agents": []}),
        run(agents=["rag"], cv_ids=[]),
        expected_cv(None),
    )

    assert record["passed"] is None
    assert record["route_ok"] is None


def test_summary_counts_runs_and_cases_that_always_pass():
    records = [
        {**score(case(), run(), expected_cv(MATCHES)), "repeat": 1},
        {**score(case(), run(cv_ids=[9]), expected_cv(MATCHES)), "repeat": 2},
        {**score(case(id="C1-02"), run(), expected_cv(MATCHES)), "repeat": 1},
        score(case(id="N-01", cv=None, expected={"report_only": True}), run(), expected_cv(None)),
    ]
    summary = summarize(records)

    assert (summary["runs"], summary["runs_passed"]) == (3, 2)
    assert (summary["cases"], summary["cases_always_passed"]) == (2, 1)
    assert summary["by_category"] == {"C1": {"runs": 3, "passed": 2}}


# --- Satu kasus lewat graph (graph palsu) ------------------------------------------


def test_run_case_records_route_ids_and_planner_tokens():
    seen = {}

    class FakeGraph:
        def invoke(self, state):
            seen["state"] = state
            cv = AgentResult(answer="- [ID:1] Job 1", source="cv", job_ids=[1, 2, 3, 4, 5])
            usage = LLMUsage(
                llm_calls=1,
                total_tokens=3400,
                calls=[LLMCall(step="planner", total_tokens=3400)],
            )
            return {
                "response": "answer",
                "subtasks": [{"agent": "cv_match", "question": "Lowongan cocok"}],
                "subresults": [cv],
                "result": cv.model_copy(update={"usage": usage}),
            }

    record = run_case(case(), FakeGraph(), {"cv01": (PROFILE, MATCHES)})

    assert seen["state"]["profile"] == PROFILE
    assert seen["state"]["matches"] == MATCHES
    assert seen["state"]["messages"] == [
        {"role": "user", "content": "Lowongan yang cocok untukku?"}
    ]
    assert record["passed"] is True
    assert record["cv_ids"] == [1, 2, 3, 4, 5]
    assert record["planner_tokens"] == 3400


def test_run_case_without_cv_sends_no_profile():
    class BrokenGraph:
        def invoke(self, state):
            assert state["profile"] is None and state["matches"] is None
            raise RuntimeError("rusak")

    record = run_case(case(cv=None, expected={"report_only": True}), BrokenGraph(), {})

    assert record["error"] == "RuntimeError: rusak"
    assert record["passed"] is None


# --- File kasus ----------------------------------------------------------------------


def test_case_file_is_consistent():
    profiles_path, cases = load_cases(CASES_PATH)
    ids = [item["id"] for item in cases]

    assert profiles_path == Path("tests/eval/cv_chat_profiles.json")
    assert len(ids) == len(set(ids)) == 45
    for item in cases:
        assert item["cv"] in {"cv01", "cv02", "cv03", "cv04", "cv05", None}
        options = [item["expected"]["agents"], *item["expected"].get("agents_alt", [])]
        assert all(set(option) <= set(AGENT_NODES) for option in options)
        assert set(item.get("checks", {})) <= {
            "standalone_all",
            "standalone_any",
            "standalone_none",
            "answer_any",
        }
    assert load_cases(CASES_PATH, ["C3"])[1][0]["id"] == "C3-01"


def test_profile_fixture_has_no_identity_and_rebuilds_on_real_database():
    profiles_path, _ = load_cases(CASES_PATH)
    data = json.loads(profiles_path.read_text(encoding="utf-8"))
    profiles = load_profiles(profiles_path, "data/database.sqlite")

    assert [case["id"] for case in data["cases"]] == ["cv01", "cv02", "cv03", "cv04", "cv05"]
    for case in data["cases"]:
        assert set(case["profile"]) <= {
            "target_role",
            "education_level",
            "education_major",
            "skills",
            "experience_months",
        }
    for profile, matches in profiles.values():
        assert profile.full_name is None and profile.contacts == []
        assert len(matches.matches) >= 5


def test_response_alt_is_accepted():
    record = score(
        case(expected={"response": "answer", "response_alt": ["unsupported"], "agents": []}),
        run(response="unsupported", agents=[]),
        {"ids": [], "gap": []},
    )

    assert record["passed"] is True
