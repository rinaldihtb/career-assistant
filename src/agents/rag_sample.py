"""Uji manual RAG agent terhadap Qdrant dan data/database.sqlite asli.

    make sample-rag       retrieval saja: embedding query + Qdrant + SQLite (tanpa LLM chat)
    make sample-rag-llm   agent lengkap: retrieval + 1 panggilan LLM + guard sitasi

Mode retrieval: kalibrasi ``DEFAULT_SCORE_THRESHOLD`` dan ``DEFAULT_CANDIDATE_K``.
Pencarian dijalankan dengan threshold 0 agar kandidat di bawah threshold tetap
terlihat, lalu setiap kandidat ditandai LOLOS atau DIBUANG terhadap threshold aktif.

Relevansi dicek tanpa LLM: setiap pertanyaan punya query SQL pembanding yang
ditulis manual (judul/lokasi), lalu dihitung berapa dari top-N yang ada di
hasil pembanding. Ini proxy kasar: judul "Digital Graphic Designer" tidak
dihitung relevan untuk "digital marketing" walaupun mungkin berguna.

Mode LLM: pertanyaan yang sama, dicatat waktu, panggilan LLM, token, dan
relevansi ID yang DIKUTIP (bukan kandidat). Kategori "tanpa hasil" dan
"di luar topik" lulus jika tidak ada ID yang dikutip. "akuntan Surabaya"
dinilai manual (aturan ditetapkan sebelum smoke test): pembandingnya hanya
ID 466 "Senior Finance & Investment Manager", yang bukan posisi akuntan, jadi
lulus jika tidak ada ID dikutip ATAU hanya 466 dengan pernyataan terus terang
bahwa posisi akuntan tidak ada.

Biaya mode retrieval: hanya embedding pertanyaan. Mode LLM menambah paling banyak
10 panggilan chat (pertanyaan tanpa kandidat tidak memanggil LLM).
"""

import os
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from src.agents.rag_agent import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_SCORE_THRESHOLD,
    retrieve_jobs,
    run_rag_agent,
)

load_dotenv()
DB_PATH = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")

# (pertanyaan, kategori, query pembanding untuk ID relevan; None = tidak ada yang relevan).
# "tanpa hasil" dan "di luar topik" idealnya 0 lowongan LOLOS.
QUESTIONS = [
    (
        "Lowongan data analyst",
        "posisi",
        "SELECT id FROM jobs WHERE lower(job_title) LIKE '%data analyst%'",
    ),
    (
        "Lowongan software engineer backend Python",
        "posisi",
        (
            "SELECT id FROM jobs WHERE lower(job_title) LIKE '%engineer%' "
            "OR lower(job_title) LIKE '%developer%' OR lower(job_title) LIKE '%python%'"
        ),
    ),
    (
        "Lowongan digital marketing di Jakarta",
        "posisi+kota",
        (
            "SELECT id FROM jobs WHERE lower(job_title) LIKE '%digital marketing%' "
            "AND lower(location) LIKE '%jakarta%'"
        ),
    ),
    (
        "Lowongan akuntan di Surabaya",
        "posisi+kota",
        (
            "SELECT id FROM jobs WHERE (lower(job_title) LIKE '%finance%' "
            "OR lower(job_title) LIKE '%akunt%' OR lower(job_title) LIKE '%account%') "
            "AND lower(location) LIKE '%surabaya%'"
        ),
    ),
    (
        "Lowongan sales di Tangerang",
        "posisi+kota",
        (
            "SELECT id FROM jobs WHERE lower(job_title) LIKE '%sales%' "
            "AND lower(location) LIKE '%tangerang%'"
        ),
    ),
    (
        "Ada lowongan apa saja di Bali?",
        "kota saja",
        "SELECT id FROM jobs WHERE lower(location) LIKE '%bali%'",
    ),
    ("Lowongan pilot pesawat komersial", "tanpa hasil", None),
    ("Lowongan dokter spesialis bedah di Medan", "tanpa hasil", None),
    ("Resep nasi goreng yang enak", "di luar topik", None),
    ("Siapa presiden Indonesia sekarang?", "di luar topik", None),
]


def _truth_ids(db_path: str | Path, sql: str | None) -> set[int]:
    """Jalankan query pembanding dengan koneksi biasa (bukan lewat agent)."""
    if sql is None:
        return set()
    with closing(sqlite3.connect(db_path)) as db:
        return {row[0] for row in db.execute(sql)}


def _location_ok(places: list[str], jobs: list[dict]) -> str:
    """Cek tanpa LLM: jika ada lokasi terdeteksi, semua lowongan harus berlokasi di sana."""
    if not places:
        return "-"
    ok = all(any(p in (job["location"] or "").lower() for p in places) for job in jobs)
    return "OK" if ok else "SALAH"


def run_retrieval_check(vector_store: VectorStore, db_path: str | Path = DB_PATH) -> None:
    """Jalankan semua pertanyaan uji dan cetak kandidat, skor, serta ringkasan metrik."""
    threshold = DEFAULT_SCORE_THRESHOLD
    summary = []
    for question, category, truth_sql in QUESTIONS:
        truth = _truth_ids(db_path, truth_sql)
        start = time.perf_counter()
        result = retrieve_jobs(
            question, vector_store=vector_store, db_path=db_path, score_threshold=0.0
        )
        elapsed = time.perf_counter() - start

        # Sama dengan aturan di retrieve_jobs: ada lokasi -> tanpa threshold.
        cutoff = float("-inf") if result.places else threshold
        passed = [job for job in result.jobs if job["score"] >= cutoff]
        relevant = sum(job["id"] in truth for job in result.jobs)
        summary.append((question, category, result, passed, relevant, truth, elapsed))

        print("=" * 100)
        print(f"Q: {question}  [{category}]")
        print(
            f"Lokasi: {result.places or '-'} | di-embed: {result.search_query!r} | "
            f"chunk {result.chunks_found} | lowongan unik {result.unique_jobs} | {elapsed:.2f}s"
        )
        print(f"Relevan menurut pembanding: {len(truth)} lowongan")
        for job in result.jobs:
            mark = "LOLOS  " if job["score"] >= cutoff else "DIBUANG"
            hit = "*" if job["id"] in truth else " "
            location = (job["location"] or "-").replace("\n", " ")
            print(
                f" {hit}{mark} {job['score']:.3f} [ID:{job['id']}] {job['job_title']} | "
                f"{job['company_name']} | {location}"
            )
            print(f"          chunk: {job['matched_chunk'][:110]!r}")

    print("\n" + "=" * 100)
    print(f"Threshold {threshold} | k {DEFAULT_CANDIDATE_K} | * = relevan menurut pembanding")
    print(
        f"{'Pertanyaan':<42} {'kategori':<13} {'unik':>4} {'lolos':>5} {'relevan':>7} "
        f"{'top1':>6} {'detik':>6}  lokasi"
    )
    for question, category, result, passed, relevant, truth, elapsed in summary:
        top1 = f"{result.jobs[0]['score']:.3f}" if result.jobs else "-"
        rel = f"{relevant}/{len(result.jobs)}" if truth else "-"
        print(
            f"{question[:42]:<42} {category:<13} {result.unique_jobs:>4} {len(passed):>5} "
            f"{rel:>7} {top1:>6} {elapsed:>6.2f}  {_location_ok(result.places, result.jobs)}"
        )
    avg = sum(item[6] for item in summary) / len(summary)
    print(f"\nRata-rata waktu {avg:.2f}s")
    print(
        "Catatan: 'unik' < 5 berarti k terlalu kecil; 'tanpa hasil'/'di luar topik' "
        "dengan lolos > 0 berarti threshold terlalu rendah; 'relevan' = top-N yang "
        "cocok dengan query pembanding."
    )


def _citation_check(result, truth: set[int]) -> str:
    """Nilai ID yang dikutip terhadap query pembanding.

    Pembanding kosong (tanpa hasil, di luar topik) berarti jawaban yang benar
    adalah tidak mengutip lowongan apa pun.
    """
    if not truth:
        return "OK (menolak)" if not result.job_ids else "SALAH (mengutip)"
    if not result.job_ids:
        return "cek manual (0 dikutip)"
    relevant = sum(job_id in truth for job_id in result.job_ids)
    return f"{relevant}/{len(result.job_ids)} relevan"


def run_llm_check(
    vector_store: VectorStore, llm: BaseChatModel, db_path: str | Path = DB_PATH
) -> None:
    """Jalankan ``run_rag_agent`` lengkap untuk semua pertanyaan uji dan cetak metriknya."""
    summary = []
    for question, category, truth_sql in QUESTIONS:
        truth = _truth_ids(db_path, truth_sql)
        start = time.perf_counter()
        result = run_rag_agent(question, llm=llm, vector_store=vector_store, db_path=db_path)
        elapsed = time.perf_counter() - start
        check = _citation_check(result, truth)
        summary.append((question, category, result, check, elapsed))

        print("=" * 100)
        print(f"Q: {question}  [{category}]")
        print(f"Jawaban:\n{result.answer}")
        print(f"Dikutip: {result.job_ids} | pembanding: {len(truth)} lowongan | {check}")
        print(f"Error  : {result.error}")
        print(
            f"Waktu {elapsed:.2f}s | LLM calls {result.usage.llm_calls} | "
            f"token in/out/total {result.usage.input_tokens}/{result.usage.output_tokens}/"
            f"{result.usage.total_tokens}"
        )

    print("\n" + "=" * 100)
    print(
        f"{'Pertanyaan':<42} {'kategori':<13} {'detik':>6} {'calls':>5} {'token':>6} "
        f"{'dikutip':>7}  cek"
    )
    for question, category, result, check, elapsed in summary:
        print(
            f"{question[:42]:<42} {category:<13} {elapsed:>6.2f} {result.usage.llm_calls:>5} "
            f"{result.usage.total_tokens:>6} {len(result.job_ids):>7}  {check}"
        )
    total_tokens = sum(item[2].usage.total_tokens for item in summary)
    avg_time = sum(item[4] for item in summary) / len(summary)
    errors = sum(item[2].error is not None for item in summary)
    print(f"\nRata-rata waktu {avg_time:.2f}s | total token {total_tokens} | error {errors}")


if __name__ == "__main__":
    from src.config.database import VECTOR_DB  # import di sini: test tidak butuh koneksi Qdrant

    if "--llm" in sys.argv:
        from src.config.llm import LLM_MODEL

        run_llm_check(VECTOR_DB, LLM_MODEL)
    else:
        run_retrieval_check(VECTOR_DB)
