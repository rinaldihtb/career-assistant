"""Retrieval RAG: repository, deteksi lokasi, dan pencarian, tanpa API key.

Vector store memakai stub dari conftest, kecuali satu test yang memakai Qdrant
in-memory asli untuk membuktikan bentuk filter ``metadata.job_id`` diterima Qdrant.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant import QdrantVectorStore

from src.agents.rag_agent import CANDIDATE_K, RagAgent
from src.repositories.job_repository import JobRepository

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


def _agent(repository: JobRepository) -> RagAgent:
    return RagAgent(llm=None, job_repository=repository)


def _retrieve(repository: JobRepository, question: str, **kwargs):
    return _agent(repository).retrieve_jobs(question, **kwargs)[1]


# --- JobRepository -----------------------------------------------------------


def test_get_jobs_by_ids_keeps_input_order(make_repository):
    jobs = make_repository().get_jobs_by_ids([5, 1, 3])
    assert [job["id"] for job in jobs] == [5, 1, 3]
    assert jobs[1]["job_title"] == "Data Analyst"


def test_get_jobs_by_ids_skips_missing_and_duplicates(make_repository):
    assert [job["id"] for job in make_repository().get_jobs_by_ids([2, 999, 2])] == [2]


def test_get_jobs_by_ids_empty_input(make_repository):
    assert make_repository().get_jobs_by_ids([]) == []


def test_get_jobs_by_ids_includes_description(make_repository):
    job = make_repository().get_jobs_by_ids([1])[0]
    assert job["job_description"] == "Analisis data penjualan dengan SQL dan Python."


def test_get_job_locations_returns_all_rows(make_repository):
    assert len(make_repository().get_job_locations()) == 6


# --- Deteksi lokasi ----------------------------------------------------------


@pytest.fixture
def agent(make_repository) -> RagAgent:
    return _agent(make_repository())


@pytest.fixture
def place_index(agent):
    return agent._build_place_index(agent.job_repository.get_job_locations())


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
def test_find_places(agent, place_index, question, expected_ids):
    assert _ids(place_index, agent._find_places(question, place_index)) == expected_ids


def test_longer_place_name_wins(agent, place_index):
    assert agent._find_places("kerja di Jakarta Selatan", place_index) == ["jakarta selatan"]


def test_place_must_be_whole_word(agent):
    index = agent._build_place_index([(1, "Serang, Banten"), (2, "Kembangan, Jakarta Raya")])
    assert agent._find_places("analis serangan siber", index) == []
    assert agent._find_places("manajer pengembangan bisnis", index) == []
    assert agent._find_places("lowongan di Serang", index) == ["serang"]


def test_place_index_is_exact_not_substring(agent):
    index = agent._build_place_index(
        [(1, "Central Karang, Banten"), (2, "Cikarang Pusat, Jawa Barat")]
    )
    assert _ids(index, agent._find_places("lowongan di Karang", index)) == {1}
    assert _ids(index, agent._find_places("lowongan di Cikarang", index)) == {2}


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
def test_remove_places(agent, question, places, expected):
    assert agent._remove_places(question, places) == expected


# --- retrieve_jobs dengan stub -----------------------------------------------


def test_dedupe_keeps_best_chunk_per_job(make_repository):
    result = _retrieve(make_repository(CHUNKS), "lowongan data analyst")

    assert [job["id"] for job in result.jobs] == [1, 2, 6]
    assert result.jobs[0]["score"] == 0.62
    assert result.jobs[0]["matched_chunk"] == "Menganalisis data penjualan memakai SQL dan Python"
    assert result.jobs[0]["company_name"] == "PT Alpha"


def test_tool_content_is_json_of_artifact(make_repository):
    content, result = _agent(make_repository(CHUNKS)).retrieve_jobs("lowongan data analyst")
    assert content == result.model_dump_json()


def test_top_n_limits_results(make_repository):
    result = _retrieve(make_repository(CHUNKS), "data", top_n=2)
    assert [job["id"] for job in result.jobs] == [1, 2]


def test_no_place_means_no_filter(make_repository):
    repository = make_repository(CHUNKS)
    result = _retrieve(repository, "lowongan data analyst")

    assert result.places == []
    assert repository.vector_db.calls == [
        {"query": "lowongan data analyst", "k": CANDIDATE_K, "filter": None}
    ]


def test_place_filter_restricts_to_place_jobs(make_repository):
    repository = make_repository(CHUNKS)
    result = _retrieve(repository, "lowongan data di Jakarta")

    call = repository.vector_db.calls[0]
    assert result.places == ["jakarta"]
    assert set(call["filter"].must[0].match.any) == {1, 4, 6}
    assert call["query"] == "lowongan data di"  # nama kota tidak ikut di-embed
    assert [job["id"] for job in result.jobs] == [1, 6]


def test_excluded_place_keeps_jobs_elsewhere(make_repository):
    repository = make_repository(CHUNKS)
    result = _retrieve(repository, "lowongan data selain di Jakarta")

    call = repository.vector_db.calls[0]
    assert (result.places, result.excluded_places) == ([], ["jakarta"])
    assert set(call["filter"].must[0].match.any) == {2, 3, 5}
    assert call["query"] == "lowongan data selain di"
    assert [job["id"] for job in result.jobs] == [2, 5, 3]  # filter aktif: tanpa threshold


@pytest.mark.parametrize(
    "question",
    [
        "Lowongan di Jawa kecuali Bandung",
        "Lowongan di Jawa di luar Bandung",
        "Jawa, bukan dari Bandung",
    ],
)
def test_requested_place_minus_excluded_place(make_repository, question):
    repository = make_repository(CHUNKS)
    result = _retrieve(repository, question)

    assert (result.places, result.excluded_places) == (["jawa"], ["bandung"])
    assert set(repository.vector_db.calls[0]["filter"].must[0].match.any) == {2}


def test_no_job_left_after_exclusion_skips_search(make_repository):
    repository = make_repository(CHUNKS)
    result = _retrieve(repository, "Lowongan di Jakarta Selatan selain Jakarta")

    assert result.jobs == []
    assert repository.vector_db.calls == []


def test_all_below_threshold_returns_empty(make_repository):
    result = _retrieve(make_repository([(1, LONG, 0.40), (2, LONG, 0.30)]), "resep nasi goreng")
    assert result.jobs == []


def test_place_filter_skips_threshold(make_repository):
    repository = make_repository([(6, LONG, 0.30), (1, LONG, 0.20), (2, LONG, 0.90)])
    result = _retrieve(repository, "Ada lowongan apa saja di Jakarta?")
    assert [job["id"] for job in result.jobs] == [6, 1]  # job 2 (Surabaya) tetap tersaring


def test_stale_vector_id_is_skipped(make_repository):
    result = _retrieve(make_repository([(999, LONG, 0.70), (1, LONG, 0.60)]), "data")
    assert [job["id"] for job in result.jobs] == [1]


def test_chunk_without_job_id_is_ignored(make_repository):
    class NoMetadataStore:
        def similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
            return [(Document(page_content=LONG), 0.9)]

    repository = make_repository()
    repository.vector_db = NoMetadataStore()
    assert _retrieve(repository, "data").jobs == []


def test_short_heading_chunk_is_ignored(make_repository):
    repository = make_repository([(2, "Deskripsi Pekerjaan", 0.80), (1, LONG, 0.60)])
    assert [job["id"] for job in _retrieve(repository, "lowongan").jobs] == [1]


# --- Qdrant in-memory asli ---------------------------------------------------


def test_job_id_filter_works_on_real_qdrant(agent):
    """Stub tidak bisa membuktikan Qdrant menerima key/tipe filter; test ini bisa."""
    docs = [
        Document(page_content=f"deskripsi {i}", metadata={"job_id": job_id, "chunk_index": i})
        for i, job_id in enumerate([1, 1, 2, 3, 4, 4, 6])
    ]
    store = QdrantVectorStore.from_documents(
        docs, DeterministicFakeEmbedding(size=16), location=":memory:", collection_name="jobs"
    )
    place_filter = agent._build_filter({1, 4, 6})
    hits = store.similarity_search_with_score("deskripsi", k=20, filter=place_filter)

    assert {doc.metadata["job_id"] for doc, _ in hits} == {1, 4, 6}
    assert len(hits) == 5
