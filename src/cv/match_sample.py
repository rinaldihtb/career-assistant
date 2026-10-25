"""Ukur Cocokkan CV dengan API asli untuk memilih konfigurasi.

    make sample-match ARGS=~/Downloads/CV.pdf

Biaya: 1 panggilan LLM (profil) + 2 embedding (ringkasan profil dan teks CV) + 6 query
Qdrant. Yang dicetak hanya angka, ID, dan judul lowongan; teks CV, nama, dan daftar skill
tidak dicetak dan tidak ada yang ditulis ke disk.

Yang diukur per konfigurasi:
- Proxy relevansi: berapa dari top-10 yang judulnya data/AI/software (``TITLE_PROXY``,
  85 dari 473 lowongan). Hanya cocok untuk CV data/AI; set eval C1d memakai gold per CV.
- Distribusi cosine semua lowongan kandidat (maks, p90, p50) untuk menentukan
  ``semantic_floor`` dan ``semantic_ceiling``.
- Kestabilan terhadap ``k``: top-10 default dengan k=800 harus sama dengan k=400.
"""

import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.cv.match import MatchConfig, load_job_skill_index, profile_query_text, rank_jobs
from src.cv.pdf_text import CVRejectedError, extract_cv_text
from src.cv.profile import extract_cv_profile
from src.cv.skills import build_cv_skills

load_dotenv()
DB_PATH = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")
ALL = 1000  # ambil semua kandidat agar distribusi cosine lengkap
SHOW = 10
TITLE_PROXY = re.compile(
    r"\b(data|ai|ml|machine learning|python|software|developer|programmer|scientist)\b",
    re.IGNORECASE,
)
CONFIGS = {
    "A default (max, prior 2)": MatchConfig(limit=ALL),
    "B max, prior 0": MatchConfig(skill_prior=0, limit=ALL),
    "C semantik saja": MatchConfig(semantic_weight=1.0, limit=ALL),
    "D teks CV, prior 2": MatchConfig(query="cv_text", limit=ALL),
    "E mean, prior 2": MatchConfig(aggregation="mean", limit=ALL),
    "A k=800": MatchConfig(k=800, limit=ALL),
}


def percentile(values: list[float], q: float) -> float:
    """Persentil sederhana (nearest-rank) dari daftar yang belum tentu terurut."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main(paths: list[str]) -> int:
    """Jalankan pengukuran untuk satu PDF; kembalikan exit code."""
    if len(paths) != 1:
        print("Pemakaian: make sample-match ARGS=path/ke/cv.pdf")
        return 2
    from src.config.database import VECTOR_DB  # import di sini: test tidak butuh koneksi Qdrant
    from src.config.llm import EMBEDDING, LLM_MODEL

    try:
        cv_text = extract_cv_text(Path(paths[0]).expanduser().read_bytes()).text
    except CVRejectedError as error:
        print(f"DITOLAK ({error.reason}): {error}")
        return 1
    started = time.perf_counter()
    result = extract_cv_profile(cv_text, llm=LLM_MODEL)
    if result.profile is None:
        print(f"Profil GAGAL: {result.error}")
        return 1
    profile = result.profile
    print(f"profil: {time.perf_counter() - started:.2f} dtk, {len(profile.skills)} skill")

    queries = {"profile": profile_query_text(profile), "cv_text": cv_text}
    vectors = {}
    for name, text in queries.items():
        started = time.perf_counter()
        vectors[name] = EMBEDDING.embed_query(text)
        print(f"embed {name}: {len(text)} karakter, {time.perf_counter() - started:.2f} dtk")

    cv_skills = build_cv_skills(profile.skills)
    print(f"skill CV: {len(cv_skills.vocab)} di kosakata, {len(cv_skills.extras)} di luar")
    started = time.perf_counter()
    load_job_skill_index(str(Path(DB_PATH).resolve()))
    print(f"indeks skill lowongan: {time.perf_counter() - started:.2f} dtk")

    baseline: list[int] = []
    for name, config in CONFIGS.items():
        started = time.perf_counter()
        matches, chunks, unique = rank_jobs(
            vectors[config.query], cv_skills, vector_store=VECTOR_DB, db_path=DB_PATH, config=config
        )
        elapsed = time.perf_counter() - started
        top = matches[:SHOW]
        ids = [m.job_id for m in top]
        baseline = baseline or ids
        cosines = [m.semantic for m in matches]
        relevant = sum(1 for m in top if TITLE_PROXY.search(m.job_title))
        print(f"\n=== {name}: {chunks} chunk, {unique} lowongan, {elapsed:.2f} dtk")
        print(f"proxy relevan top-{SHOW}: {relevant}/{SHOW}")
        print(
            f"cosine kandidat maks/p90/p50: {max(cosines):.3f} / "
            f"{percentile(cosines, 0.9):.3f} / {percentile(cosines, 0.5):.3f}"
        )
        print(f"top-{SHOW} sama dengan A: {len(set(ids) & set(baseline))}/{SHOW}")
        for i, m in enumerate(top, start=1):
            mark = "*" if TITLE_PROXY.search(m.job_title) else " "
            required = len(m.matched_skills) + len(m.missing_skills)
            print(
                f"{i:2}.{mark}#{m.job_id:<3} {m.score:4.0%} cos={m.semantic:.3f} "
                f"skill={m.skill_score:.2f} ({len(m.matched_skills)}/{required})  "
                f"{m.job_title[:45]} | {(m.company_name or '-')[:25]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
