"""Ukur Cocokkan CV dengan API asli untuk memilih konfigurasi.

    make sample-match ARGS=~/Downloads/CV.pdf

Biaya: 1 panggilan LLM (profil) + 2 embedding (ringkasan profil dan teks CV) + 5 query
Qdrant. Yang dicetak hanya angka, ID, dan judul lowongan; teks CV, nama, dan daftar skill
tidak dicetak dan tidak ada yang ditulis ke disk.

Yang diukur:
- 4 konfigurasi (ringkasan profil vs teks CV) x (mean top-3 vs max), plus k=800 untuk
  memeriksa apakah k=400 sudah cukup (top-10 harus sama).
- Distribusi skor cosine hasil agregasi (peringkat 1, 10, 50) untuk menentukan
  ``semantic_floor`` dan ``semantic_ceiling``.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from cv.match import MatchConfig, load_job_skill_index, profile_query_text, rank_jobs
from cv.pdf_text import CVRejectedError, extract_cv_text
from cv.profile import extract_cv_profile
from cv.skills import build_cv_skills

load_dotenv()
DB_PATH = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")
STATS_LIMIT = 50
SHOW = 10
CONFIGS = {
    "A profil+mean": MatchConfig(limit=STATS_LIMIT),
    "B profil+max": MatchConfig(aggregation="max", limit=STATS_LIMIT),
    "C cvtext+mean": MatchConfig(query="cv_text", limit=STATS_LIMIT),
    "D cvtext+max": MatchConfig(query="cv_text", aggregation="max", limit=STATS_LIMIT),
    "A k=800": MatchConfig(k=800, limit=STATS_LIMIT),
}


def main(paths: list[str]) -> int:
    """Jalankan pengukuran untuk satu PDF; kembalikan exit code."""
    if len(paths) != 1:
        print("Pemakaian: make sample-match ARGS=path/ke/cv.pdf")
        return 2
    from config.database import VECTOR_DB  # import di sini: test tidak butuh koneksi Qdrant
    from config.llm import EMBEDDING, LLM_MODEL

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
        raw = [m.semantic for m in matches]
        ids = [m.job_id for m in matches[:SHOW]]
        baseline = baseline or ids
        rank = {i: f"{raw[i - 1]:.3f}" if len(raw) >= i else "-" for i in (1, 10, 50)}
        print(f"\n=== {name}: {chunks} chunk, {unique} lowongan, {elapsed:.2f} dtk")
        print(f"cosine agregat peringkat 1/10/50: {rank[1]} / {rank[10]} / {rank[50]}")
        print(f"top-{SHOW} sama dengan A: {len(set(ids) & set(baseline))}/{SHOW}")
        no_skill = sum(1 for m in matches[:SHOW] if m.skill_portion is None)
        print(f"top-{SHOW} tanpa skill kosakata: {no_skill}")
        for i, m in enumerate(matches[:SHOW], start=1):
            portion = "-" if m.skill_portion is None else f"{m.skill_portion:.2f}"
            required = len(m.matched_skills) + len(m.missing_skills)
            print(
                f"{i:2}. #{m.job_id:<3} {m.score:4.0%} cos={m.semantic:.3f} "
                f"skill={portion} ({len(m.matched_skills)}/{required})  "
                f"{m.job_title[:45]} | {(m.company_name or '-')[:25]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
