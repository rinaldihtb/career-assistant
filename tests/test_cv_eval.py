"""Test set eval Cocokkan CV: penilaian profil, gold, kalibrasi, grid, isi set eval, dan
kesamaan urutan dengan ``rank_jobs``. Tanpa API.
"""

import json
import re
from pathlib import Path

import pytest
from langchain_core.documents import Document

from cv.match import MatchConfig, load_job_skill_index, rank_jobs
from cv.pdf_text import extract_cv_text
from cv.profile import Achievement, CVProfile, CVProfileResult, Experience
from cv.skills import build_cv_skills
from evaluation.run_cv_eval import (
    AGGREGATIONS,
    CASES_PATH,
    QUERIES,
    SEMANTIC_WEIGHTS,
    SKILL_PRIORS,
    TOP_N,
    analyze,
    calibrate,
    check_achievements,
    check_profile,
    check_skills,
    choose_config,
    collect_candidates,
    evaluate_config,
    gold_job_ids,
    load_cases,
    load_titles,
    precision_at,
    quantile,
    rank_candidates,
    run_case,
    same_skill,
    squash,
)

REAL_DB = Path(__file__).resolve().parents[1] / "data" / "database.sqlite"
FILLER = " Detail tanggung jawab dan kualifikasi untuk posisi ini."
# Job 1 SQL + Python, job 5 REST API + Python, job 3 dan 6 tanpa skill CV, job 4 chunk pendek,
# job 99 ID basi (tidak ada di SQLite).
CHUNKS = [
    (1, "Analisis data penjualan." + FILLER, 0.60),
    (1, "Membuat laporan mingguan." + FILLER, 0.55),
    (2, "Membuat dashboard." + FILLER, 0.58),
    (3, "Membuat materi visual." + FILLER, 0.45),
    (5, "Membangun REST API." + FILLER, 0.62),
    (6, "Rekrutmen karyawan." + FILLER, 0.40),
    (4, "Pendek", 0.95),
    (99, "Lowongan yang sudah dihapus." + FILLER, 0.70),
]


class VectorStoreByVector:
    """Stub ``similarity_search_with_score_by_vector``; skor chunk ditentukan test."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def similarity_search_with_score_by_vector(self, embedding, k=4, **kwargs):
        self.calls += 1
        hits = [
            (Document(page_content=text, metadata={"job_id": job_id}), score)
            for job_id, text, score in self.chunks
        ]
        hits.sort(key=lambda hit: hit[1], reverse=True)
        return hits[:k]


class FakeEmbeddings:
    """Embedding palsu: mencatat teks yang di-embed."""

    def __init__(self, error=None):
        self.error = error
        self.texts = []

    def embed_query(self, text):
        if self.error:
            raise self.error
        self.texts.append(text)
        return [0.1, 0.2, 0.3]


def candidate(job_id, cos, matched=0, required=0, mean=None):
    return {
        "job_id": job_id,
        "max": cos,
        "mean": cos if mean is None else mean,
        "matched": matched,
        "required": required,
    }


def row(mean_p5, mean_p10, min_p5, *, selectable=True, **config):
    base = {
        "query": "profile",
        "aggregation": "max",
        "skill_prior": 2.0,
        "semantic_weight": 0.6,
        "semantic_floor": 0.25,
        "semantic_ceiling": 0.65,
    }
    base.update(config)
    return {
        "config": base,
        "selectable": selectable,
        "mean_p5": mean_p5,
        "mean_p10": mean_p10,
        "min_p5": min_p5,
    }


# ------------------------------------------------------------------ kandidat dan urutan


def test_collect_candidates_filters_like_rank_jobs(jobs_db):
    index = load_job_skill_index(str(jobs_db.resolve()))
    store = VectorStoreByVector(CHUNKS)
    candidates = collect_candidates(
        [0.1], build_cv_skills(["Python", "SQL"]), vector_store=store, index=index
    )
    assert store.calls == 1
    assert [c["job_id"] for c in candidates] == [1, 2, 3, 5, 6]  # tanpa 4 (pendek) dan 99
    first = candidates[0]
    assert first["max"] == pytest.approx(0.60)
    assert first["mean"] == pytest.approx((0.60 + 0.55 + 0.40) / 3)  # isian chunk terendah
    assert (first["matched"], first["required"]) == (2, 2)
    job5 = candidates[3]
    assert (job5["matched"], job5["required"]) == (1, 2)  # Python cocok, REST API kurang


@pytest.mark.parametrize(
    "config",
    [
        MatchConfig(),
        MatchConfig(aggregation="mean"),
        MatchConfig(skill_prior=0.0, semantic_weight=0.4),
        MatchConfig(semantic_floor=0.5, semantic_ceiling=0.6, semantic_weight=0.8),
    ],
)
def test_rank_candidates_matches_rank_jobs(jobs_db, config):
    """Grid hanya sahih jika urutan hitung ulang sama persis dengan aplikasi."""
    cv_skills = build_cv_skills(["Python", "SQL"])
    index = load_job_skill_index(str(jobs_db.resolve()))
    candidates = collect_candidates(
        [0.1], cv_skills, vector_store=VectorStoreByVector(CHUNKS), index=index
    )
    matches, _, _ = rank_jobs(
        [0.1], cv_skills, vector_store=VectorStoreByVector(CHUNKS), db_path=jobs_db, config=config
    )
    ranked = rank_candidates(candidates, config)
    assert [job_id for job_id, _ in ranked] == [m.job_id for m in matches]
    assert [score for _, score in ranked] == pytest.approx([m.score for m in matches])


def test_precision_at_counts_missing_positions_as_irrelevant():
    assert precision_at([1, 2, 3, 4, 5], {1, 3, 9}, 5) == pytest.approx(0.4)
    assert precision_at([1, 2], {1, 2}, 5) == pytest.approx(0.4)


def fake_profile(monkeypatch, profile):
    calls = []

    def extract(cv_text, *, llm, today=None):
        calls.append(cv_text)
        return CVProfileResult(profile=profile)

    monkeypatch.setattr("evaluation.run_cv_eval.extract_cv_profile", extract)
    return calls


def test_run_case_one_llm_two_embeddings_two_searches(monkeypatch, jobs_db):
    llm_calls = fake_profile(
        monkeypatch, CVProfile(target_role="Data Analyst", skills=["Python", "SQL"])
    )
    embeddings = FakeEmbeddings()
    store = VectorStoreByVector(CHUNKS)
    case = {"id": "cv01", "file": "tests/eval/cv/cv01_data_analyst.pdf"}

    record, cv_text = run_case(
        case, llm=None, embeddings=embeddings, vector_store=store, db_path=jobs_db
    )

    assert record["error"] is None
    assert len(llm_calls) == 1
    assert embeddings.texts == ["Data Analyst\nSkills: Python, SQL", cv_text]
    assert store.calls == 2
    assert set(record["candidates"]) == {"profile", "cv_text"}
    assert "Dokumen fiktif" not in json.dumps(record)  # teks CV tidak ikut disimpan


def test_run_case_reports_errors_without_raising(monkeypatch, jobs_db):
    fake_profile(monkeypatch, CVProfile(skills=["Python"]))
    case = {"id": "cv01", "file": "tests/eval/cv/cv01_data_analyst.pdf"}
    record, _ = run_case(
        case,
        llm=None,
        embeddings=FakeEmbeddings(error=RuntimeError("403")),
        vector_store=VectorStoreByVector(CHUNKS),
        db_path=jobs_db,
    )
    assert record["error"] == "Embedding/Qdrant error: 403"

    missing = {"id": "cv09", "file": "tests/eval/cv/tidak_ada.pdf"}
    record, cv_text = run_case(
        missing, llm=None, embeddings=FakeEmbeddings(), vector_store=None, db_path=jobs_db
    )
    assert record["error"].startswith("PDF tidak terbaca")
    assert cv_text is None


# ------------------------------------------------------------------ gold dan profil


def test_gold_job_ids_include_and_exclude():
    titles = {
        1: "Data Analyst",
        2: "Data Engineer",
        3: "Junior Graphic Designer (AI-Assisted Design)",
        4: "HRD (KPI Specialist - People Analytics)",
        5: "Software Engineer",
    }
    case = {"title_include": r"\bdata\b|analytics", "title_exclude": r"graphic|\bhrd?\b"}
    assert gold_job_ids(case, titles) == [1, 2]
    assert gold_job_ids({"title_include": "engineer", "title_exclude": ""}, titles) == [2, 5]
    with_extra = {**case, "extra_ids": [5, 99]}  # 99 tidak ada di database: diabaikan
    assert gold_job_ids(with_extra, titles) == [1, 2, 5]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("Rekrutmen", "Rekrutmen & Seleksi", True),
        ("Microsoft Excel (VBA)", "excel", True),
        ("  power   BI ", "Power BI", True),
        ("Excel", "Excellent communication", False),
        ("Java", "JavaScript", False),
        ("", "SQL", False),
    ],
)
def test_same_skill(a, b, expected):
    assert same_skill(a, b) is expected


def test_check_skills_reports_missed_invented_and_extra():
    cv_text = "Skills: SQL, Python, Rekrutmen & Seleksi\nLanguages: Indonesian, English"
    result = check_skills(
        ["SQL", "Rekrutmen", "English", "Indonesian", "Docker"],
        ["SQL", "Python", ["Inggris", "English"], "Rekrutmen & Seleksi"],
        cv_text,
    )
    assert (result["found"], result["gold"]) == (3, 4)
    assert result["recall"] == pytest.approx(0.75)
    assert result["missed"] == ["Python"]
    assert result["invented"] == ["Docker"]  # tidak ada di teks CV
    assert result["extra"] == ["Indonesian"]  # ada di teks, di luar gold


def test_check_profile():
    profile = CVProfile(
        education_level="S1",
        experiences=[Experience(title="HR Staff", company=None, start="2020", end="2021")],
    )
    result = check_profile(profile, {"education_level": "S1", "experiences": 2})
    assert result["education_level"] == {"expected": "S1", "actual": "S1"}
    assert result["experiences"] == {"expected": 2, "actual": 1}


ACHIEVEMENT_CV = (
    "Teknisi Listrik – PT Daya Prima, Gresik\n"
    "\uf0b7 Merawat 14 panel distribusi dan menurunkan gangguan\nbulanan sebesar 30%.\n"
    "P\nE\nN\nG\nH\nA\nR\nG\nA\nA\nN\n"
    "Juara 2 lomba instalasi tingkat provinsi (2021)."
)


def test_squash_ignores_line_breaks_bullets_and_dash_style():
    assert squash("Teknisi  Listrik – PT\n\uf0b7 Daya") == "teknisilistrik-ptdaya"


def test_check_achievements_accepts_quotes_across_line_breaks():
    achievements = [
        Achievement(
            text="Merawat 14 panel distribusi dan menurunkan gangguan bulanan sebesar 30%.",
            context="Teknisi Listrik - PT Daya Prima",
        ),
        Achievement(text="Juara 2 lomba instalasi tingkat provinsi (2021).", context="PENGHARGAAN"),
        Achievement(text="menurunkan gangguan bulanan", context=None),
    ]
    result = check_achievements(achievements, ACHIEVEMENT_CV)
    assert result == {"count": 3, "numeric": 2, "not_verbatim": [], "context_unknown": 0}


def test_check_achievements_flags_changed_numbers_translation_and_unknown_context():
    achievements = [
        Achievement(text="Merawat 15 panel distribusi", context="Teknisi Listrik, PT Daya Prima"),
        Achievement(text="Maintained 14 distribution panels", context="Supervisor, PT Daya Prima"),
    ]
    result = check_achievements(achievements, ACHIEVEMENT_CV)
    assert result["not_verbatim"] == [
        "Merawat 15 panel distribusi",
        "Maintained 14 distribution panels",
    ]
    assert result["context_unknown"] == 1  # "Supervisor" tidak ada di teks CV


# ------------------------------------------------------------------ kalibrasi dan pilihan


def test_quantile_interpolates():
    assert quantile([4, 1, 3, 2], 0.5) == pytest.approx(2.5)
    assert quantile([4, 1, 3, 2], 1.0) == 4
    assert quantile([4, 1, 3, 2], 0.0) == 1


def test_calibrate_lowest_median_and_highest_top_rounded_down():
    sets = [
        [candidate(1, 0.40), candidate(2, 0.52), candidate(3, 0.707)],  # median 0.52
        [candidate(4, 0.45), candidate(5, 0.494), candidate(6, 0.66)],  # median 0.494
        [],
    ]
    assert calibrate(sets, "max") == (0.45, 0.70)
    assert calibrate(sets, "max", step=0.01) == (0.49, 0.70)
    assert calibrate([[candidate(1, 0.70)]], "max") == (0.70, 0.75)  # ceiling > floor
    assert calibrate([[candidate(1, 0.30)], [candidate(2, 0.35)]], "max") == (0.30, 0.35)


def test_evaluate_config_precision_and_top_score():
    candidates = [candidate(i, 0.60 - i / 100) for i in range(1, 13)]
    runs = [{"id": "cv01", "gold": {1, 2, 3, 11}, "candidates": {"profile": candidates}}]
    config = MatchConfig(semantic_weight=1.0, semantic_floor=0.40, semantic_ceiling=0.60)
    result = evaluate_config(config, runs)
    assert result["p5"] == {"cv01": pytest.approx(0.6)}
    assert result["p10"] == {"cv01": pytest.approx(0.3)}
    assert result["top_score"]["cv01"] == pytest.approx(0.95)


def test_choose_config_tie_breaks_then_prefers_default():
    rows = [
        row(0.9, 0.9, 0.8, selectable=False),  # pembanding tidak ikut dipilih
        row(0.8, 0.7, 0.6),
        row(0.8, 0.8, 0.4, skill_prior=0.0),
        row(0.8, 0.8, 0.4, semantic_weight=0.7),
        row(0.8, 0.8, 0.4, aggregation="mean"),
    ]
    assert choose_config(rows)["config"]["semantic_weight"] == 0.7
    rows.append(row(0.8, 0.8, 0.6, query="cv_text"))
    assert choose_config(rows)["config"]["query"] == "cv_text"  # p@5 terburuk lebih tinggi


def test_analyze_end_to_end_without_api():
    def record(cv_id, gold, shift):
        cands = [
            candidate(i, 0.62 - i / 100 - shift, matched=i % 3, required=3) for i in range(1, 15)
        ]
        return {
            "id": cv_id,
            "error": None,
            "gold_ids": gold,
            "candidates": {"profile": cands, "cv_text": cands[::-1]},
            "skills_check": {"recall": 1.0, "invented": []},
            "profile_check": {"experiences": {"expected": 2, "actual": 2}},
        }

    records = [
        record("cv01", [1, 2, 3], 0.0),
        record("cv02", [4, 5], 0.05),
        {"id": "cv03", "error": "Profil gagal"},
    ]
    titles = {i: f"Job {i}" for i in range(1, 15)}
    summary = analyze(records, titles)

    assert summary["cases_scored"] == 2
    assert set(summary["calibration"]) == {f"{q}/{a}" for q in QUERIES for a in AGGREGATIONS}
    per_query = len(SKILL_PRIORS) * len(SEMANTIC_WEIGHTS) + 1
    assert len(summary["grid"]) == len(QUERIES) * len(AGGREGATIONS) * per_query
    assert summary["chosen"]["selectable"] is True
    assert len(summary["top"]["cv01"]) == TOP_N
    assert summary["top"]["cv01"][0]["title"].startswith("Job")
    assert summary["profile_fields_ok"] == 2
    unjudged = summary["unjudged"]["cv01"]
    assert unjudged and not set(unjudged) & {1, 2, 3}  # non-gold tanpa penilaian manual


def test_analyze_marks_judged_non_gold():
    cands = [candidate(i, 0.70 - i / 100, matched=1, required=2) for i in range(1, 12)]
    record = {
        "id": "cv01",
        "error": None,
        "gold_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "not_relevant_ids": [9, 10],
        "candidates": {"profile": cands, "cv_text": cands},
    }
    summary = analyze([record], {})
    assert summary["unjudged"] == {}
    assert [item["judged"] for item in summary["top"]["cv01"]] == [True] * TOP_N


def test_analyze_without_scored_cases():
    assert analyze([{"id": "cv01", "error": "x"}], {})["error"]


# ------------------------------------------------------------------ isi set eval


CASES = load_cases(CASES_PATH)


def test_cases_file_is_consistent():
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids)) == 5
    for case in CASES:
        re.compile(case["title_include"])
        re.compile(case["title_exclude"])
        assert case["profile"]["skills"]
        extra = set(case["extra_ids"])
        not_relevant = set(case["not_relevant_ids"])
        assert not extra & not_relevant
        assert len(extra) == len(case["extra_ids"])


@pytest.mark.skipif(not REAL_DB.is_file(), reason="data/database.sqlite tidak tersedia")
def test_manual_judgements_are_outside_title_gold():
    """extra_ids hanya untuk judul yang tidak tertangkap regex; not_relevant_ids bukan gold."""
    titles = load_titles(REAL_DB)
    for case in CASES:
        by_title = set(gold_job_ids({**case, "extra_ids": []}, titles))
        assert not set(case["extra_ids"]) & by_title, case["id"]
        assert not set(case["not_relevant_ids"]) & set(gold_job_ids(case, titles)), case["id"]
        assert set(case["extra_ids"]) <= set(titles), case["id"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_cv_pdf_is_fictional_readable_and_contains_gold_skills(case):
    """Gold skill harus benar-benar tertulis di CV; PDF harus berlabel fiktif."""
    text = extract_cv_text(Path(case["file"]).read_bytes()).text
    assert "Dokumen fiktif untuk evaluasi" in text
    normalized = " ".join(text.casefold().split())
    for item in case["profile"]["skills"]:
        names = [item] if isinstance(item, str) else item
        assert any(name.casefold() in normalized for name in names), item


@pytest.mark.parametrize(
    ("cv_id", "markers"),
    [
        (
            "cv02",
            [
                "CONTACT",
                "SOFTWARE",
                "After Effects",
                "SKILLS",
                "Social Media Content",
                "LANGUAGES",
                "PROFILE",
                "EXPERIENCE",
                "Junior Graphic Designer",
                "EDUCATION",
            ],
        ),
        (
            "cv04",
            [
                "Professional Summary",
                "Work Experience",
                "Sales Staff - PT Cahaya",
                "Education",
                "Skills",
                "Marketing, Sales, Communication",
                "Languages",
            ],
        ),
    ],
)
def test_two_column_cv_columns_do_not_mix(cv_id, markers):
    case = next(case for case in CASES if case["id"] == cv_id)
    text = extract_cv_text(Path(case["file"]).read_bytes()).text
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)


@pytest.mark.skipif(not REAL_DB.is_file(), reason="data/database.sqlite tidak tersedia")
def test_gold_counts_match_real_database():
    titles = load_titles(REAL_DB)
    counts = {case["id"]: len(gold_job_ids(case, titles)) for case in CASES}
    assert counts == {case["id"]: case["gold_count"] for case in CASES}
