"""Skill gap dari lowongan teratas: Python murni, tanpa LLM dan tanpa database."""

from src.cv.gap import GAP_LIMIT, SkillCount, compute_skill_gap
from src.cv.match import JobMatch


def job(job_id: int, score: float, matched: list[str], missing: list[str], **extra) -> JobMatch:
    data = {
        "job_id": job_id,
        "job_title": f"Job {job_id}",
        "score": score,
        "semantic": 0.6,
        "semantic_scaled": 0.6,
        "skill_portion": 0.5,
        "matched_skills": matched,
        "missing_skills": missing,
    }
    data.update(extra)
    return JobMatch(**data)


MATCHES = [
    job(1, 0.85, ["SQL", "Excel"], ["Power BI", "dbt"]),
    job(2, 0.80, ["SQL"], ["Power BI", "Statistik"]),
    job(3, 0.70, ["SQL", "Excel"], ["Looker Studio"]),
    job(4, 0.65, [], [], skill_portion=None),
]


def names(counts: tuple[SkillCount, ...]) -> list[tuple[str, int]]:
    return [(count.name, count.jobs) for count in counts]


def test_missing_ordered_by_job_count_then_job_score():
    gap = compute_skill_gap(MATCHES)

    # Power BI 2 lowongan; sisanya 1 lowongan, urut skor lowongan: dbt 0,85 > Statistik 0,80.
    assert names(gap.missing) == [
        ("Power BI", 2),
        ("dbt", 1),
        ("Statistik", 1),
        ("Looker Studio", 1),
    ]
    assert gap.missing[0].score_sum == 0.85 + 0.80


def test_have_counts_matched_skills():
    gap = compute_skill_gap(MATCHES)

    assert names(gap.have) == [("SQL", 3), ("Excel", 2)]


def test_counts_jobs_and_jobs_without_skills():
    gap = compute_skill_gap(MATCHES)

    assert (gap.jobs, gap.jobs_without_skills) == (4, 1)
    assert (gap.missing_total, gap.have_total) == (4, 2)


def test_equal_count_and_score_keeps_first_seen_order():
    # missing_skills per lowongan sudah urut frekuensi di semua lowongan; urutan itu dipakai.
    gap = compute_skill_gap([job(1, 0.8, [], ["Excel", "VBA", "Access"])])

    assert [count.name for count in gap.missing] == ["Excel", "VBA", "Access"]


def test_duplicate_skill_in_one_job_counted_once():
    gap = compute_skill_gap([job(1, 0.8, ["SQL", "SQL"], ["dbt", "dbt"])])

    assert names(gap.have) == [("SQL", 1)]
    assert names(gap.missing) == [("dbt", 1)]


def test_limit_cuts_list_but_keeps_total():
    missing = [f"Skill {i}" for i in range(GAP_LIMIT + 3)]
    gap = compute_skill_gap([job(1, 0.8, [], missing)])

    assert len(gap.missing) == GAP_LIMIT
    assert gap.missing_total == GAP_LIMIT + 3
    assert len(compute_skill_gap([job(1, 0.8, [], missing)], limit=3).missing) == 3


def test_empty_matches():
    gap = compute_skill_gap([])

    assert (gap.missing, gap.have, gap.jobs, gap.jobs_without_skills) == ((), (), 0, 0)
