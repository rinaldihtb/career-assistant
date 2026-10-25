"""Retrieval RAG agent (Tahap 4): repository, deteksi lokasi, dan pencarian.

Semua test berjalan tanpa API key. Vector store memakai stub dari conftest,
kecuali satu test yang memakai Qdrant in-memory asli untuk membuktikan
bentuk filter ``metadata.job_id`` diterima Qdrant.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant import QdrantVectorStore

from src.agents.rag_agent import (
    allowed_job_ids,
    build_job_id_filter,
    build_place_index,
    detect_places,
    remove_places,
    retrieve_jobs,
    split_excluded_places,
)
from src.repositories.job_repository import get_job_locations, get_jobs_by_ids

# --- job_repository ----------------------------------------------------------


def test_get_jobs_by_ids_keeps_input_order(jobs_db):
    jobs = get_jobs_by_ids(jobs_db, [5, 1, 3])
    assert [job["id"] for job in jobs] == [5, 1, 3]
    assert jobs[1]["job_title"] == "Data Analyst"


def test_get_jobs_by_ids_skips_missing_and_duplicates(jobs_db):
    jobs = get_jobs_by_ids(jobs_db, [2, 999, 2])
    assert [job["id"] for job in jobs] == [2]


def test_get_jobs_by_ids_empty_input_does_not_open_db(tmp_path):
    assert get_jobs_by_ids(tmp_path / "tidak_ada.sqlite", []) == []


def test_get_jobs_by_ids_excludes_seeding_timestamps(jobs_db):
    job = get_jobs_by_ids(jobs_db, [1])[0]
    assert "created_at" not in job
    assert "job_description" in job


def test_get_job_locations_returns_all_rows(jobs_db):
    assert len(get_job_locations(jobs_db)) == 6


# --- Deteksi lokasi ----------------------------------------------------------


@pytest.fixture
def place_index(jobs_db):
    return build_place_index(get_job_locations(jobs_db))


def _ids(place_index, places):
    return set().union(*(place_index[key] for key in places)) if places else set()


@pytest.mark.parametrize(
    ("question", "expected_ids"),
    [
        ("Lowongan data analyst di Jakarta", {1, 4, 6}),  # 4 = "Jakarta Raya\n(Hibrid)"
        ("lowongan di JAKARTA SELATAN", {1}),
        ("Lowongan di Jawa", {2, 3}),
        ("Lowongan di Jawa Barat", {3}),
        ("Surabaya atau Bandung?", {2, 3}),
        ("Lowongan backend engineer", set()),
    ],
)
def test_detect_places(place_index, question, expected_ids):
    assert _ids(place_index, detect_places(question, place_index)) == expected_ids


def test_longer_place_name_wins(place_index):
    assert detect_places("kerja di Jakarta Selatan", place_index) == ["jakarta selatan"]


def test_place_must_be_whole_word():
    index = build_place_index([(1, "Serang, Banten"), (2, "Kembangan, Jakarta Raya")])
    assert detect_places("analis serangan siber", index) == []
    assert detect_places("manajer pengembangan bisnis", index) == []
    assert detect_places("lowongan di Serang", index) == ["serang"]


def test_place_index_is_exact_not_substring():
    index = build_place_index([(1, "Central Karang, Banten"), (2, "Cikarang Pusat, Jawa Barat")])
    assert _ids(index, detect_places("lowongan di Karang", index)) == {1}
    assert _ids(index, detect_places("lowongan di Cikarang", index)) == {2}


# --- retrieve_jobs dengan stub -----------------------------------------------

# Teks chunk dibuat >= MIN_CHUNK_CHARS (40) agar tidak terbuang sebagai judul bagian.
CHUNKS = [
    (1, "Menganalisis data penjualan memakai SQL dan Python", 0.62),
    (1, "Membuat laporan mingguan untuk tim manajemen", 0.58),  # chunk kedua job 1: dibuang
    (2, "Membantu cleaning data dan membuat dashboard harian", 0.55),
    (6, "Mengelola rekrutmen dan administrasi karyawan", 0.47),
    (5, "Membangun REST API dengan Python dan PostgreSQL", 0.44),  # di bawah threshold 0,45
    (3, "Membuat materi visual untuk kampanye media sosial", 0.30),
]
LONG = "teks chunk yang cukup panjang untuk lolos batas minimal"


def test_dedupe_keeps_best_chunk_per_job(jobs_db, fake_vector_store):
    store = fake_vector_store(CHUNKS)
    result = retrieve_jobs("lowongan data analyst", vector_store=store, db_path=jobs_db)

    assert [job["id"] for job in result.jobs] == [1, 2, 6]
    assert result.jobs[0]["score"] == 0.62
    assert result.jobs[0]["matched_chunk"] == "Menganalisis data penjualan memakai SQL dan Python"
    assert result.jobs[0]["company_name"] == "PT Alpha"
    assert result.chunks_found == 6
    assert result.unique_jobs == 5
    assert result.below_threshold == 2


@pytest.mark.parametrize(
    ("question", "places", "expected"),
    [
        ("Lowongan selain di Jakarta", ["jakarta"], ([], ["jakarta"])),
        ("loker di luar jakarta", ["jakarta"], ([], ["jakarta"])),
        ("Lowongan admin kecuali Bandung", ["bandung"], ([], ["bandung"])),
        ("Lowongan bukan dari Surabaya", ["surabaya"], ([], ["surabaya"])),
        ("Lowongan di Jakarta", ["jakarta"], (["jakarta"], [])),
        (
            "Lowongan di Surabaya, selain Jakarta",
            ["surabaya", "jakarta"],
            (["surabaya"], ["jakarta"]),
        ),
    ],
)
def test_split_excluded_places(question, places, expected):
    assert split_excluded_places(question, places) == expected


def test_allowed_job_ids_subtracts_excluded(place_index):
    assert allowed_job_ids(place_index, [], ["jakarta"]) == {2, 3, 5}  # semua yang berlokasi
    assert allowed_job_ids(place_index, ["jawa"], ["bandung"]) == {2}
    assert allowed_job_ids(place_index, ["jakarta"], []) == {1, 4, 6}


def test_excluded_place_filters_out_that_city(jobs_db, fake_vector_store):
    store = fake_vector_store(CHUNKS)
    result = retrieve_jobs("lowongan data selain di Jakarta", vector_store=store, db_path=jobs_db)

    assert result.places == []
    assert result.excluded_places == ["jakarta"]
    assert set(store.calls[0]["filter"].must[0].match.any) == {2, 3, 5}
    assert all("jakarta" not in job["location"].lower() for job in result.jobs)


def test_exclusion_covering_requested_place_returns_nothing(jobs_db, fake_vector_store):
    store = fake_vector_store(CHUNKS)
    result = retrieve_jobs(
        "lowongan di Jakarta Selatan selain Jakarta", vector_store=store, db_path=jobs_db
    )

    assert result.jobs == []
    assert store.calls == []  # tidak ada kandidat, Qdrant tidak dipanggil


def test_top_n_limits_results(jobs_db, fake_vector_store):
    store = fake_vector_store(CHUNKS)
    result = retrieve_jobs("data", vector_store=store, db_path=jobs_db, top_n=2)
    assert [job["id"] for job in result.jobs] == [1, 2]


def test_no_place_means_no_filter(jobs_db, fake_vector_store):
    store = fake_vector_store(CHUNKS)
    result = retrieve_jobs("lowongan data analyst", vector_store=store, db_path=jobs_db)

    assert result.places == []
    assert result.search_query == "lowongan data analyst"
    assert store.calls == [{"query": "lowongan data analyst", "k": 20, "filter": None}]


def test_place_filter_restricts_to_place_jobs(jobs_db, fake_vector_store):
    store = fake_vector_store(CHUNKS)
    result = retrieve_jobs("lowongan data di Jakarta", vector_store=store, db_path=jobs_db)

    assert result.places == ["jakarta"]
    assert set(store.calls[0]["filter"].must[0].match.any) == {1, 4, 6}
    assert store.calls[0]["query"] == "lowongan data di"  # nama kota tidak ikut di-embed
    assert result.search_query == "lowongan data di"
    assert [job["id"] for job in result.jobs] == [1, 6]


def test_all_below_threshold_returns_empty(jobs_db, fake_vector_store):
    store = fake_vector_store([(1, LONG, 0.40), (2, LONG, 0.30)])
    result = retrieve_jobs("resep nasi goreng", vector_store=store, db_path=jobs_db)

    assert result.jobs == []
    assert result.below_threshold == 2


def test_stale_vector_id_is_skipped(jobs_db, fake_vector_store):
    store = fake_vector_store([(999, LONG, 0.70), (1, LONG, 0.60)])
    result = retrieve_jobs("data", vector_store=store, db_path=jobs_db)
    assert [job["id"] for job in result.jobs] == [1]


def test_chunk_without_job_id_is_ignored(jobs_db):
    class NoMetadataStore:
        def similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
            return [(Document(page_content=LONG), 0.9)]

    result = retrieve_jobs("data", vector_store=NoMetadataStore(), db_path=jobs_db)
    assert result.jobs == []
    assert result.chunks_found == 1


def test_short_heading_chunk_is_ignored(jobs_db, fake_vector_store):
    store = fake_vector_store([(2, "Deskripsi Pekerjaan", 0.80), (1, LONG, 0.60)])
    result = retrieve_jobs("lowongan", vector_store=store, db_path=jobs_db)

    assert [job["id"] for job in result.jobs] == [1]
    assert result.chunks_found == 2
    assert result.unique_jobs == 1


@pytest.mark.parametrize(
    ("question", "places", "expected"),
    [
        ("Lowongan akuntan di Surabaya", ["surabaya"], "Lowongan akuntan di"),
        ("Sales di Jakarta Selatan atau Tangerang?", ["jakarta selatan", "tangerang"],
         "Sales di atau ?"),
        ("Bali", ["bali"], "Bali"),  # sisa kosong: pakai pertanyaan asli
        ("Lowongan di Bali?", ["bali"], "Lowongan di ?"),
    ],
)  # fmt: skip
def test_remove_places(question, places, expected):
    assert remove_places(question, places) == expected


# --- Qdrant in-memory asli ---------------------------------------------------


def test_job_id_filter_works_on_real_qdrant():
    """Stub tidak bisa membuktikan Qdrant menerima key/tipe filter; test ini bisa."""
    docs = [
        Document(page_content=f"deskripsi {i}", metadata={"job_id": job_id, "chunk_index": i})
        for i, job_id in enumerate([1, 1, 2, 3, 4, 4, 6])
    ]
    store = QdrantVectorStore.from_documents(
        docs, DeterministicFakeEmbedding(size=16), location=":memory:", collection_name="jobs"
    )
    hits = store.similarity_search_with_score(
        "deskripsi", k=20, filter=build_job_id_filter({1, 4, 6})
    )

    assert {doc.metadata["job_id"] for doc, _ in hits} == {1, 4, 6}
    assert len(hits) == 5


def test_place_filter_skips_threshold(jobs_db, fake_vector_store):
    store = fake_vector_store([(6, LONG, 0.30), (1, LONG, 0.20), (2, LONG, 0.90)])
    result = retrieve_jobs("Ada lowongan apa saja di Jakarta?", vector_store=store, db_path=jobs_db)

    assert [job["id"] for job in result.jobs] == [6, 1]  # job 2 (Surabaya) tetap tersaring
    assert result.below_threshold == 0
