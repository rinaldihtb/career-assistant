"""RAG agent: mencari lowongan yang relevan lalu menjawab dengan sitasi [ID:x].

Alur ``run_rag_agent`` (0 atau 1 panggilan LLM chat):
    1. ``retrieve_jobs`` (tanpa LLM chat, hanya embedding query):
       a. Deteksi lokasi dengan kamus nama tempat dari kolom ``location`` SQLite;
          lokasi setelah "selain/kecuali/di luar/bukan" dianggap pengecualian.
       b. Qdrant ``k`` chunk; jika ada lokasi, difilter ``metadata.job_id`` (MatchAny:
          lokasi yang diminta dikurangi lokasi yang dikecualikan) dan nama tempat
          dibuang dari teks yang di-embed.
       c. Buang chunk sangat pendek, dedupe per ``job_id`` (chunk skor tertinggi),
          buang skor < threshold (hanya jika tidak ada filter lokasi), ambil ``top_n``.
       d. Detail lowongan dari SQLite berdasarkan ID.
    2. Tanpa kandidat: pesan tetap, 0 panggilan LLM.
    3. ``answer_from_jobs``: satu panggilan LLM menjawab dari kandidat, lalu guard
       sitasi tanpa LLM menyaring ID yang dikutip.

``retrieve_jobs`` terpisah dari ``answer_from_jobs`` sehingga retrieval bisa diukur tanpa
LLM chat (``make sample-rag``).
"""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field
from qdrant_client import models

from agents.prompts import (
    RAG_ANSWER_SYSTEM,
    RAG_ANSWER_USER,
    RAG_FAILED_MESSAGE,
    RAG_JOB_TEMPLATE,
    RAG_NO_CANDIDATE_MESSAGE,
)
from models.agent import AgentResult, LLMUsage
from repositories.job_repository import get_job_locations, get_jobs_by_ids

DEFAULT_TOP_N = 5
DEFAULT_CANDIDATE_K = 20  # chunk yang diambil dari Qdrant sebelum dedupe per lowongan
# Hanya berlaku untuk pertanyaan tanpa lokasi. Diukur dengan make sample-rag: top-1 relevan
# 0,64-0,72, terlemah yang relevan 0,497; di luar topik maks 0,444; "pilot" 0,452.
# Sengaja tidak dinaikkan ke 0,46 hanya demi satu titik data; LLM di
# answer_from_jobs adalah lapisan kedua yang menolak lowongan tidak relevan.
DEFAULT_SCORE_THRESHOLD = 0.45
# Chunk lebih pendek dari ini hampir selalu judul bagian ("About the role",
# "Deskripsi Pekerjaan"): 73 dari 3.613 chunk. Isinya generik sehingga mirip
# dengan pertanyaan apa pun yang mengandung kata "lowongan".
MIN_CHUNK_CHARS = 40
JOB_ID_METADATA_KEY = "metadata.job_id"  # payload langchain_qdrant: {"metadata": {...}}
# Deskripsi lowongan rata-rata jauh lebih panjang; 800 karakter pertama biasanya
# sudah memuat ringkasan posisi. Bersama matched_chunk (bagian yang paling cocok)
# ini cukup untuk menilai relevansi dengan token kecil: 5 kandidat ~1.500 token.
MAX_DESCRIPTION_CHARS = 800
# Longgar terhadap variasi penulisan LLM: "[ID:12]", "[ID: 12]", "[ID:12, ID:15]".
_CITATION = re.compile(r"\bID:\s*(\d+)", re.IGNORECASE)

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
        places: Kunci lokasi yang diminta di pertanyaan (huruf kecil).
        excluded_places: Kunci lokasi yang dikecualikan ("selain Jakarta" -> ``["jakarta"]``).
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
    excluded_places: list[str] = Field(default_factory=list)
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


# Kata sebelum nama tempat yang membalik maknanya: "selain Jakarta", "di luar Bandung",
# "kecuali di Bali", "bukan dari Surabaya". Tanpa ini, filter lokasi justru membatasi
# kandidat ke kota yang ingin dihindari.
EXCLUSION_PREFIX = r"(?:selain|kecuali|di\s+luar|bukan)\s+(?:(?:di|dari)\s+)?"


def split_excluded_places(question: str, places: Iterable[str]) -> tuple[list[str], list[str]]:
    """Pisahkan kunci tempat menjadi ``(diminta, dikecualikan)``.

    Kenapa regex, bukan LLM: pengecualian lokasi di pertanyaan pengguna hampir
    selalu memakai empat kata ini tepat sebelum nama tempat, jadi pencocokan
    teks cukup, gratis, dan bisa dites. Batasan yang disengaja: hanya tempat
    yang langsung didahului kata pengecualian; "selain Jakarta dan Bandung"
    hanya mengecualikan Jakarta.
    """
    included: list[str] = []
    excluded: list[str] = []
    for key in places:
        words = r"\s+".join(map(re.escape, key.split()))
        pattern = re.compile(rf"(?<!\w){EXCLUSION_PREFIX}{words}(?!\w)", re.IGNORECASE)
        (excluded if pattern.search(question) else included).append(key)
    return included, excluded


def allowed_job_ids(
    place_index: dict[str, set[int]], included: Iterable[str], excluded: Iterable[str]
) -> set[int]:
    """ID lowongan yang lolos filter lokasi: lokasi diminta dikurangi lokasi dikecualikan.

    Jika hanya ada pengecualian ("lowongan selain di Jakarta"), titik awalnya
    semua lowongan yang punya lokasi.
    """
    included = list(included)
    keys = included or list(place_index)
    allowed = set().union(*(place_index[key] for key in keys))
    return allowed - set().union(*(place_index[key] for key in excluded))


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


# --- Entry point -------------------------------------------------------------


def run_rag_agent(
    question: str,
    *,
    llm: BaseChatModel,
    vector_store: VectorStore,
    db_path: str | Path,
    top_n: int = DEFAULT_TOP_N,
) -> AgentResult:
    """Cari lowongan yang relevan dan jawab pertanyaan dengan sitasi [ID:x].

    Jumlah panggilan LLM chat:
        - ada kandidat: 1
        - tidak ada kandidat (semua di bawah threshold): 0, jawaban pesan tetap
        - retrieval gagal (Qdrant, embedding, SQLite): 0, jawaban pesan tetap

    Args:
        question: Pertanyaan mandiri dari supervisor.
        llm: Chat model yang disuntikkan (asli atau palsu saat test).
        vector_store: Vector store yang disuntikkan (``QdrantVectorStore`` atau stub).
        db_path: Path ke file SQLite.
        top_n: Jumlah kandidat maksimal yang dikirim ke LLM.

    Returns:
        ``AgentResult`` dengan ``source="rag"``. ``job_ids`` dan ``rows`` hanya
        berisi lowongan yang dikutip LLM DAN memang ada di kandidat. Fungsi ini
        tidak pernah melempar exception; kegagalan dicatat di ``error``.
    """
    try:
        retrieval = retrieve_jobs(question, vector_store=vector_store, db_path=db_path, top_n=top_n)
    except Exception as exc:  # noqa: BLE001 - kontrak: agent tidak pernah melempar exception
        return AgentResult(answer=RAG_FAILED_MESSAGE, source="rag", error=f"Retrieval error: {exc}")

    if not retrieval.jobs:
        return AgentResult(answer=RAG_NO_CANDIDATE_MESSAGE, source="rag")

    return answer_from_jobs(question, retrieval.jobs, llm=llm)


# --- Jawaban dengan LLM ------------------------------------------------------


def answer_from_jobs(
    question: str, jobs: list[dict[str, Any]], *, llm: BaseChatModel
) -> AgentResult:
    """Satu panggilan LLM menjawab dari kandidat, lalu guard sitasi tanpa LLM.

    Kenapa guard di Python, bukan meminta LLM mengembalikan JSON daftar ID:
    jawaban teks dan ID yang dikutip tidak mungkin berbeda karena ID diambil
    langsung dari teks jawaban. Tidak perlu panggilan kedua untuk verifikasi.

    Args:
        question: Pertanyaan mandiri.
        jobs: Kandidat dari ``retrieve_jobs`` (dict kolom lowongan + ``matched_chunk``).
        llm: Chat model.

    Returns:
        ``AgentResult`` dengan ``source="rag"``. Tidak pernah melempar exception.
    """
    usage = LLMUsage()
    user_prompt = RAG_ANSWER_USER.format(
        question=question, count=len(jobs), jobs="\n\n".join(map(format_job_for_prompt, jobs))
    )
    try:
        response = llm.invoke([SystemMessage(RAG_ANSWER_SYSTEM), HumanMessage(user_prompt)])
    except Exception as exc:  # noqa: BLE001 - tipe error provider LLM sangat beragam
        return AgentResult(
            answer=RAG_FAILED_MESSAGE, source="rag", usage=usage, error=f"LLM error: {exc}"
        )
    usage.add(response, step="rag_answer")
    answer = response.text

    valid_ids, invalid_ids = check_citations(answer, [job["id"] for job in jobs])
    jobs_by_id = {job["id"]: job for job in jobs}
    return AgentResult(
        answer=answer,
        source="rag",
        job_ids=valid_ids,
        rows=[jobs_by_id[job_id] for job_id in valid_ids],
        usage=usage,
        error=f"ID di luar kandidat dibuang: {invalid_ids}" if invalid_ids else None,
    )


def check_citations(answer: str, candidate_ids: Iterable[int]) -> tuple[list[int], list[int]]:
    """Pisahkan ID yang dikutip jawaban menjadi valid (ada di kandidat) dan palsu.

    Guard ini mencegah UI menampilkan kartu lowongan yang tidak pernah dilihat
    LLM (ID halusinasi). Jawaban teks sengaja tidak diubah: dengan temperature 0
    dan ID tertulis jelas di prompt, ID palsu jarang terjadi, dan jika terjadi
    dicatat di ``AgentResult.error`` untuk dievaluasi.

    Returns:
        ``(valid, invalid)``, masing-masing urut kemunculan pertama di jawaban
        dan tanpa duplikat.
    """
    candidates = set(candidate_ids)
    cited = list(dict.fromkeys(int(match) for match in _CITATION.findall(answer)))
    valid = [job_id for job_id in cited if job_id in candidates]
    invalid = [job_id for job_id in cited if job_id not in candidates]
    return valid, invalid


def format_job_for_prompt(job: dict[str, Any]) -> str:
    """Ubah satu kandidat menjadi blok ``<job id="...">`` untuk prompt.

    Teks ``</job>`` di dalam data dinetralkan agar deskripsi lowongan tidak
    bisa menutup tag lebih awal dan menyisipkan teks di luar blok data.
    """
    description = job.get("job_description") or ""
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = f"{description[:MAX_DESCRIPTION_CHARS]}..."
    fields = {
        "job_title": job.get("job_title") or "-",
        "company_name": job.get("company_name") or "-",
        "location": " ".join((job.get("location") or "-").split()),
        "work_type": job.get("work_type") or "-",
        "salary": format_salary(job.get("salary_min"), job.get("salary_max")),
        "matched_chunk": " ".join((job.get("matched_chunk") or "-").split()),
        "description": " ".join(description.split()) or "-",
    }
    safe = {key: value.replace("</job>", "</ job>") for key, value in fields.items()}
    return RAG_JOB_TEMPLATE.format(id=job["id"], **safe)


def format_salary(salary_min: int | None, salary_max: int | None) -> str:
    """Format gaji Rupiah per bulan; ``"tidak dicantumkan"`` jika kosong.

    Contoh: ``(8_000_000, 10_000_000)`` -> ``"Rp8.000.000 - Rp10.000.000 per bulan"``.
    Teks eksplisit "tidak dicantumkan" lebih aman daripada menghapus baris Gaji:
    LLM tidak tergoda menebak angka.
    """

    def rupiah(value: int) -> str:
        return f"Rp{value:,}".replace(",", ".")

    if not salary_min and not salary_max:
        return "tidak dicantumkan"
    if salary_min and salary_max and salary_min != salary_max:
        return f"{rupiah(salary_min)} - {rupiah(salary_max)} per bulan"
    return f"{rupiah(salary_min or salary_max)} per bulan"


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
            Kandidat sudah dipersempit filter; relevansi dinilai LLM di
            ``answer_from_jobs``.

    Returns:
        ``RetrievalResult``. ``jobs`` kosong berarti tidak ada lowongan relevan.

    Raises:
        Exception: Error Qdrant, embedding, atau SQLite diteruskan apa adanya.
            ``run_rag_agent`` yang menangkapnya agar kontrak "tidak pernah
            melempar exception" tetap terjaga.
    """
    place_index = build_place_index(get_job_locations(db_path))
    detected = detect_places(question, place_index)
    places, excluded = split_excluded_places(question, detected)
    qdrant_filter = None
    search_query = question
    if detected:
        job_ids = allowed_job_ids(place_index, places, excluded)
        if not job_ids:
            return RetrievalResult(places=places, excluded_places=excluded, search_query=question)
        qdrant_filter = build_job_id_filter(job_ids)
        search_query = remove_places(question, detected)

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
    if qdrant_filter is not None:
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
        excluded_places=excluded,
        search_query=search_query,
        jobs=jobs,
        chunks_found=len(chunks),
        unique_jobs=len(best),
        below_threshold=len(ranked) - len(passed),
    )
