"""Test Cocokkan CV: agregasi, kalibrasi, skor gabungan, dan kontrak ``match_jobs``.

Tanpa API: embedding dan Qdrant diganti stub; database memakai fixture ``jobs_db``.
"""

import pytest
from langchain_core.documents import Document

from cv.match import (
    MatchConfig,
    aggregate_job_scores,
    combine_score,
    match_jobs,
    profile_query_text,
    scale_semantic,
    skill_score,
)
from cv.profile import CVProfile, Experience
from repositories.job_repository import get_job_descriptions

FILLER = " Detail tanggung jawab dan kualifikasi untuk posisi ini."


class FakeEmbeddings:
    """Embedding palsu: mencatat teks yang di-embed, vektornya tidak dipakai stub."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.texts: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        if self.error:
            raise self.error
        self.texts.append(text)
        return [0.1, 0.2, 0.3]


class VectorStoreByVector:
    """Stub ``similarity_search_with_score_by_vector``; skor chunk ditentukan test."""

    def __init__(self, chunks: list[tuple[int, str, float]], error: Exception | None = None):
        self.chunks = chunks
        self.error = error
        self.calls: list[dict] = []

    def similarity_search_with_score_by_vector(self, embedding, k=4, **kwargs):
        self.calls.append({"embedding": embedding, "k": k})
        if self.error:
            raise self.error
        hits = [
            (Document(page_content=text, metadata={"job_id": job_id}), score)
            for job_id, text, score in self.chunks
        ]
        hits.sort(key=lambda hit: hit[1], reverse=True)
        return hits[:k]


def make_profile(**overrides) -> CVProfile:
    data = {
        "full_name": "BUDI CONTOH",
        "target_role": "Data Analyst",
        "location": "Surabaya",
        "education_level": "S1",
        "education_major": "Statistika",
        "skills": ["Python", "SQL"],
        "experience_months": 12,
        "experiences": [
            Experience(title="Data Intern", company="PT A", start="2024-01", end="2024-12")
        ],
    }
    data.update(overrides)
    return CVProfile(**data)


# job 1: SQL + Python; job 5: REST API + Python; job 3: tanpa skill kosakata; 99: ID basi.
CHUNKS = [
    (1, "Analisis data penjualan." + FILLER, 0.60),
    (1, "Membuat laporan mingguan." + FILLER, 0.55),
    (1, "Kolaborasi lintas tim." + FILLER, 0.50),
    (5, "Membangun REST API." + FILLER, 0.62),
    (3, "Membuat materi visual." + FILLER, 0.30),
    (99, "Lowongan yang sudah dihapus." + FILLER, 0.70),
    (4, "Deskripsi", 0.95),
]


def test_get_job_descriptions_returns_all_rows(jobs_db):
    rows = get_job_descriptions(jobs_db)
    assert len(rows) == 6
    assert rows[0] == (1, "Analisis data penjualan dengan SQL dan Python.")


def test_profile_query_text_excludes_name_and_location():
    text = profile_query_text(make_profile())
    assert text == (
        "Data Analyst\nSkills: Python, SQL\nExperience: Data Intern\nEducation: S1 Statistika"
    )
    assert "BUDI" not in text
    assert "Surabaya" not in text


def test_profile_query_text_empty_profile():
    assert profile_query_text(CVProfile()) == ""


def test_aggregate_max():
    hits = [(1, 0.6), (1, 0.5), (2, 0.4)]
    assert aggregate_job_scores(hits, method="max") == {1: 0.6, 2: 0.4}


def test_aggregate_mean_pads_missing_chunks_with_lowest_retrieved_score():
    hits = [(1, 0.6), (1, 0.5), (1, 0.4), (1, 0.1), (2, 0.6), (3, 0.2)]
    result = aggregate_job_scores(hits, method="mean", top_chunks=3)
    assert result[1] == pytest.approx(0.5)
    assert result[2] == pytest.approx((0.6 + 0.1 + 0.1) / 3)
    assert result[3] == pytest.approx((0.2 + 0.1 + 0.1) / 3)


def test_aggregate_empty():
    assert aggregate_job_scores([]) == {}


def test_scale_semantic_is_linear_and_clipped():
    assert scale_semantic(0.45, 0.25, 0.65) == pytest.approx(0.5)
    assert scale_semantic(0.10, 0.25, 0.65) == 0.0
    assert scale_semantic(0.90, 0.25, 0.65) == 1.0
    with pytest.raises(ValueError):
        scale_semantic(0.5, 0.6, 0.6)


@pytest.mark.parametrize(
    ("matched", "required", "prior", "expected"),
    [
        (1, 1, 2.0, 1 / 3),
        (6, 7, 2.0, 6 / 9),
        (0, 0, 2.0, 0.0),
        (1, 1, 0.0, 1.0),
        (0, 0, 0.0, 0.0),
    ],
)
def test_skill_score_with_prior(matched, required, prior, expected):
    assert skill_score(matched, required, prior) == pytest.approx(expected)


def test_one_generic_skill_no_longer_beats_many_matched_skills():
    """Kegagalan terukur di CV asli: lowongan sales 1/1 ("English") mengalahkan data 6/7."""
    one_of_one = combine_score(0.5, skill_score(1, 1, 2.0), 0.6)
    six_of_seven = combine_score(0.5, skill_score(6, 7, 2.0), 0.6)
    assert six_of_seven > one_of_one
    assert combine_score(0.5, skill_score(1, 1, 0.0), 0.6) > six_of_seven  # tanpa prior


def test_combine_score_is_weighted_average():
    assert combine_score(0.5, 1.0, 0.6) == pytest.approx(0.7)
    assert combine_score(0.4, 0.0, 1.0) == pytest.approx(0.4)


def test_match_jobs_ranks_by_combined_score(jobs_db):
    embeddings = FakeEmbeddings()
    store = VectorStoreByVector(CHUNKS)
    result = match_jobs(make_profile(), embeddings=embeddings, vector_store=store, db_path=jobs_db)

    assert result.error is None
    assert result.embedding_calls == 1
    assert result.chunks_found == 7
    assert result.unique_jobs == 4  # chunk pendek job 4 dibuang; ID 99 ikut dihitung
    assert [m.job_id for m in result.matches] == [1, 5, 3]

    first, second, third = result.matches
    assert first.semantic == pytest.approx(0.60)  # default max: chunk terbaik
    assert first.semantic_scaled == pytest.approx(0.875)
    assert first.skill_portion == 1.0
    assert first.skill_score == pytest.approx(2 / 4)
    assert first.score == pytest.approx(0.6 * 0.875 + 0.4 * 0.5)
    assert first.matched_skills == ["Python", "SQL"]  # Python disebut 2 lowongan, SQL 1
    assert first.job_title == "Data Analyst"
    assert first.salary_min == 8_000_000

    assert second.skill_portion == 0.5
    assert second.skill_score == pytest.approx(1 / 4)
    assert second.missing_skills == ["REST API"]
    assert third.skill_portion is None
    assert third.skill_score == 0.0
    assert third.score == pytest.approx(0.6 * third.semantic_scaled)


def test_match_jobs_mean_aggregation(jobs_db):
    result = match_jobs(
        make_profile(),
        embeddings=FakeEmbeddings(),
        vector_store=VectorStoreByVector(CHUNKS),
        db_path=jobs_db,
        config=MatchConfig(aggregation="mean"),
    )
    assert result.matches[0].job_id == 1
    assert result.matches[0].semantic == pytest.approx(0.55)


def test_match_jobs_embeds_profile_summary_once_and_passes_k(jobs_db):
    embeddings = FakeEmbeddings()
    store = VectorStoreByVector(CHUNKS)
    match_jobs(
        make_profile(),
        embeddings=embeddings,
        vector_store=store,
        db_path=jobs_db,
        config=MatchConfig(k=50, limit=2),
    )
    assert embeddings.texts == [profile_query_text(make_profile())]
    assert store.calls == [{"embedding": [0.1, 0.2, 0.3], "k": 50}]


def test_match_jobs_limit(jobs_db):
    result = match_jobs(
        make_profile(),
        embeddings=FakeEmbeddings(),
        vector_store=VectorStoreByVector(CHUNKS),
        db_path=jobs_db,
        config=MatchConfig(limit=1),
    )
    assert [m.job_id for m in result.matches] == [1]


def test_match_jobs_cv_text_mode(jobs_db):
    embeddings = FakeEmbeddings()
    result = match_jobs(
        make_profile(),
        embeddings=embeddings,
        vector_store=VectorStoreByVector(CHUNKS),
        db_path=jobs_db,
        cv_text="  teks CV lengkap  ",
        config=MatchConfig(query="cv_text"),
    )
    assert result.error is None
    assert embeddings.texts == ["teks CV lengkap"]


def test_match_jobs_cv_text_mode_without_text_returns_error(jobs_db):
    embeddings = FakeEmbeddings()
    result = match_jobs(
        make_profile(),
        embeddings=embeddings,
        vector_store=VectorStoreByVector(CHUNKS),
        db_path=jobs_db,
        config=MatchConfig(query="cv_text"),
    )
    assert result.error is not None
    assert embeddings.texts == []


def test_match_jobs_empty_profile_makes_no_embedding_call(jobs_db):
    embeddings = FakeEmbeddings()
    result = match_jobs(
        CVProfile(),
        embeddings=embeddings,
        vector_store=VectorStoreByVector(CHUNKS),
        db_path=jobs_db,
    )
    assert result.error == "Profil atau teks CV kosong"
    assert result.embedding_calls == 0
    assert embeddings.texts == []


def test_match_jobs_embedding_error_is_reported(jobs_db):
    result = match_jobs(
        make_profile(),
        embeddings=FakeEmbeddings(error=RuntimeError("403")),
        vector_store=VectorStoreByVector(CHUNKS),
        db_path=jobs_db,
    )
    assert result.error == "Embedding error: 403"
    assert result.embedding_calls == 0
    assert result.matches == []


def test_match_jobs_vector_store_error_is_reported(jobs_db):
    result = match_jobs(
        make_profile(),
        embeddings=FakeEmbeddings(),
        vector_store=VectorStoreByVector(CHUNKS, error=ConnectionError("qdrant down")),
        db_path=jobs_db,
    )
    assert result.error == "Pencarian error: qdrant down"
    assert result.embedding_calls == 1
