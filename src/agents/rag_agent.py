"""RAG agent: mencari lowongan yang relevan dengan pertanyaan pengguna.

Tahap 4 (isi file saat ini): retrieval, tanpa LLM chat.
    1. Deteksi lokasi tanpa LLM: kamus nama tempat dari kolom ``location`` SQLite.
    2. Qdrant ``k`` chunk; jika ada lokasi, difilter ``metadata.job_id`` (MatchAny)
       dan nama tempat dibuang dari teks yang di-embed.
    3. Buang chunk sangat pendek, dedupe per ``job_id`` (chunk skor tertinggi),
       buang skor < threshold (hanya jika TIDAK ada filter lokasi), ambil ``top_n``.
    4. Detail lowongan dari SQLite berdasarkan ID.

Tahap 5 menambahkan ``run_rag_agent``: satu panggilan LLM untuk menjawab dan
guard sitasi [ID:x] tanpa LLM.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field
from qdrant_client import models

from repositories.job_repository import get_job_locations, get_jobs_by_ids

DEFAULT_TOP_N = 5
DEFAULT_CANDIDATE_K = 20  # chunk yang diambil dari Qdrant sebelum dedupe per lowongan
# Hanya berlaku untuk pertanyaan TANPA lokasi. Smoke test 2 (12 Okt): top-1 relevan
# 0,64-0,72, terlemah yang relevan 0,497; di luar topik maks 0,444; "pilot" 0,452.
# Sengaja tidak dinaikkan ke 0,46 hanya demi satu titik data; LLM di Tahap 5
# adalah lapisan kedua yang menolak lowongan tidak relevan.
DEFAULT_SCORE_THRESHOLD = 0.45
# Chunk lebih pendek dari ini hampir selalu judul bagian ("About the role",
# "Deskripsi Pekerjaan"): 73 dari 3.613 chunk. Isinya generik sehingga mirip
# dengan pertanyaan apa pun yang mengandung kata "lowongan".
MIN_CHUNK_CHARS = 40
JOB_ID_METADATA_KEY = "metadata.job_id"  # payload langchain_qdrant: {"metadata": {...}}

# Kata yang dibuang dari nama tempat untuk membuat alias pendek, misalnya
# "Jakarta Selatan" -> "jakarta", "Kabupaten Bogor" -> "bogor",
# "South Tangerang" -> "tangerang". Nama lengkap tetap ikut menjadi kunci.
GENERIC_PLACE_WORDS = frozenset(
    {
        "selatan", "utara", "barat", "timur", "pusat", "tengah", "raya",
        "kota", "kabupaten", "district", "regency",
        "south", "north", "west", "east", "central",
        "lama", "baru", "di",
    }
)  # fmt: skip


class RetrievalResult(BaseModel):
    """Hasil pencarian lowongan sebelum dijawab LLM.

    Attributes:
        places: Kunci lokasi yang terdeteksi di pertanyaan (huruf kecil).
        search_query: Teks yang benar-benar di-embed (pertanyaan tanpa nama tempat
            jika filter lokasi aktif).
        jobs: Lowongan terpilih, urut skor tertinggi. Setiap dict berisi kolom
            ``JOB_DETAIL_COLUMNS`` ditambah ``score`` dan ``matched_chunk``.
        chunks_found: Jumlah chunk yang dikembalikan Qdrant.
        unique_jobs: Jumlah lowongan unik di antara chunk tersebut.
        below_threshold: Jumlah lowongan unik yang dibuang karena skornya di bawah
            threshold. Selalu 0 jika filter lokasi aktif.
    """

    places: list[str] = Field(default_factory=list)
    search_query: str = ""
    jobs: list[dict[str, Any]] = Field(default_factory=list)
    chunks_found: int = 0
    unique_jobs: int = 0
    below_threshold: int = 0


# --- Deteksi lokasi (tanpa LLM) ----------------------------------------------


def _place_names(location: str) -> list[str]:
    """Pecah ``"Kota, Provinsi\\n(Hibrid)"`` menjadi ``["kota", "provinsi"]``."""
    first_line = location.split("\n", 1)[0]
    return [part.strip().lower() for part in first_line.split(",") if part.strip()]


def _place_keys(name: str) -> set[str]:
    """Nama lengkap plus alias tanpa kata generik, contoh ``{"jakarta selatan", "jakarta"}``."""
    keys = {name}
    short = " ".join(word for word in name.split() if word not in GENERIC_PLACE_WORDS)
    if short:
        keys.add(short)
    return keys


def build_place_index(job_locations: Iterable[tuple[int, str | None]]) -> dict[str, set[int]]:
    """Bangun kamus ``kunci tempat -> ID lowongan`` dari kolom location.

    Kenapa kamus, bukan LLM: nama tempat di data hanya 93 variasi, jadi
    pencocokan teks sudah cukup akurat, gratis, deterministik, dan bisa dites.

    Kenapa kunci dipetakan ke ID (bukan dipakai sebagai ``LIKE '%kunci%'``):
    pemetaan ini presisi. Dengan LIKE, kunci "karang" (dari "Central Karang")
    ikut mencocokkan "Cikarang", dan "riau" mencocokkan "Kepulauan Riau".

    Args:
        job_locations: Pasangan ``(id, location)``, biasanya dari ``get_job_locations``.

    Returns:
        Dict kunci huruf kecil ke himpunan ID lowongan.
    """
    index: dict[str, set[int]] = {}
    for job_id, location in job_locations:
        if not location:
            continue
        for name in _place_names(location):
            for key in _place_keys(name):
                index.setdefault(key, set()).add(int(job_id))
    return index


def _place_pattern(key: str) -> re.Pattern[str]:
    """Regex satu kunci tempat dengan batas kata; spasi di kunci cocok dengan spasi apa pun."""
    words = r"\s+".join(map(re.escape, key.split()))
    return re.compile(rf"(?<!\w){words}(?!\w)", re.IGNORECASE)


def detect_places(question: str, place_index: dict[str, set[int]]) -> list[str]:
    """Cari kunci tempat yang disebut di pertanyaan.

    Kunci terpanjang dicocokkan lebih dulu lalu dihapus dari teks, sehingga
    "Jakarta Selatan" hanya menghasilkan ``"jakarta selatan"``, bukan juga
    ``"jakarta"``. Pencocokan memakai batas kata: "serangan" tidak dianggap
    menyebut "Serang".

    Returns:
        Kunci yang ditemukan, urut dari yang terpanjang. Kosong jika tidak ada.
    """
    text = question.lower()
    found: list[str] = []
    for key in sorted(place_index, key=len, reverse=True):
        pattern = _place_pattern(key)
        if pattern.search(text):
            found.append(key)
            text = pattern.sub(" ", text)
    return found


def remove_places(question: str, places: Iterable[str]) -> str:
    """Buang nama tempat dari pertanyaan sebelum di-embed.

    Kenapa: lokasi sudah dijamin oleh filter Qdrant. Jika nama kota ikut
    di-embed, chunk yang sekadar menyebut kota ("Penempatan: Tangerang")
    mengalahkan chunk yang relevan dengan posisi. Terbukti di smoke test:
    "akuntan di Surabaya" mengembalikan 5 lowongan sales/marketing.

    Returns:
        Pertanyaan tanpa nama tempat. Jika sisanya kosong (pertanyaan hanya
        berisi nama tempat), pertanyaan asli dikembalikan karena embedding
        teks kosong tidak bermakna.
    """
    text = question
    for key in places:
        text = _place_pattern(key).sub(" ", text)
    text = " ".join(text.split())
    return text if text.strip(" ?!.,") else question


def build_job_id_filter(job_ids: Iterable[int]) -> models.Filter:
    """Filter Qdrant: hanya chunk dengan ``metadata.job_id`` di daftar ID.

    MatchAny pada ``metadata.job_id`` sudah terbukti berfungsi di koleksi
    ``jobs`` tanpa payload index; dengan 3.613 point, full scan payload masih cepat.
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key=JOB_ID_METADATA_KEY, match=models.MatchAny(any=sorted(set(job_ids)))
            )
        ]
    )


# --- Retrieval ---------------------------------------------------------------


def retrieve_jobs(
    question: str,
    *,
    vector_store: VectorStore,
    db_path: str | Path,
    top_n: int = DEFAULT_TOP_N,
    k: int = DEFAULT_CANDIDATE_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> RetrievalResult:
    """Cari lowongan paling relevan untuk pertanyaan, tanpa LLM chat.

    Satu-satunya panggilan berbayar adalah embedding query di dalam
    ``vector_store.similarity_search_with_score``.

    Args:
        question: Pertanyaan mandiri dari supervisor.
        vector_store: Vector store yang disuntikkan (``QdrantVectorStore`` asli
            atau stub saat test). Skor diasumsikan cosine similarity:
            makin besar makin relevan.
        db_path: Path ke file SQLite.
        top_n: Jumlah lowongan maksimal yang dikembalikan.
        k: Jumlah chunk yang diambil dari Qdrant sebelum dedupe.
        score_threshold: Lowongan dengan skor chunk terbaik di bawah nilai ini dibuang.
            Diabaikan jika filter lokasi aktif: pertanyaan seperti "ada lowongan
            apa di Bali?" tidak punya topik untuk dicocokkan, sehingga skornya
            rendah (smoke test: maks 0,448) walaupun semua kandidat valid.
            Kandidat sudah dipersempit filter; relevansi dinilai LLM di Tahap 5.

    Returns:
        ``RetrievalResult``. ``jobs`` kosong berarti tidak ada lowongan relevan.

    Raises:
        Exception: Error Qdrant, embedding, atau SQLite diteruskan apa adanya.
            ``run_rag_agent`` (Tahap 5) yang menangkapnya agar kontrak
            "tidak pernah melempar exception" tetap terjaga.
    """
    place_index = build_place_index(get_job_locations(db_path))
    places = detect_places(question, place_index)
    qdrant_filter = None
    search_query = question
    if places:
        place_job_ids = set().union(*(place_index[key] for key in places))
        qdrant_filter = build_job_id_filter(place_job_ids)
        search_query = remove_places(question, places)

    chunks = vector_store.similarity_search_with_score(search_query, k=k, filter=qdrant_filter)

    # Qdrant mengembalikan chunk urut skor menurun, jadi kemunculan pertama
    # setiap job_id adalah chunk terbaiknya.
    best: dict[int, tuple[float, str]] = {}
    for doc, score in chunks:
        if len(doc.page_content.strip()) < MIN_CHUNK_CHARS:
            continue
        job_id = doc.metadata.get("job_id")
        if job_id is not None and int(job_id) not in best:
            best[int(job_id)] = (float(score), doc.page_content)

    ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
    if places:
        passed = ranked
    else:
        passed = [(job_id, hit) for job_id, hit in ranked if hit[0] >= score_threshold]
    selected = passed[:top_n]

    jobs = get_jobs_by_ids(db_path, [job_id for job_id, _ in selected])
    hits = dict(selected)
    for job in jobs:
        job["score"], job["matched_chunk"] = hits[job["id"]]

    return RetrievalResult(
        places=places,
        search_query=search_query,
        jobs=jobs,
        chunks_found=len(chunks),
        unique_jobs=len(best),
        below_threshold=len(ranked) - len(passed),
    )
