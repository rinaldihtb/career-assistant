"""Ukur rencana belajar Skill Gap dengan API asli.

    make sample-plan
    make sample-plan ARGS=~/Downloads/CV.pdf

Tanpa ARGS: 5 CV fiktif di tests/eval/cv/. Per CV: ``ingest_cv`` (1 LLM) + ``match_jobs``
(1 embedding) + rencana (1 LLM), sekitar Rp15 per CV.

Yang dicetak hanya angka dan nama file: teks CV, nama, skill, dan isi rencana tidak dicetak,
dan tidak ada yang ditulis ke disk. Metrik utama ``di_luar_gap``: jumlah skill di rencana yang
tidak ada di daftar skill gap (aturan 3 prompt ``CV_PLAN_SYSTEM``). Kode penyaring skill hanya
ditambahkan jika angka ini terukur lebih dari 0.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.cv.gap import compute_skill_gap
from src.cv.ingest import ingest_cv
from src.cv.match import match_jobs
from src.cv.plan import PLAN_WEEKS, generate_learning_plan

load_dotenv()
DB_PATH = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")
DEFAULT_DIR = Path("tests/eval/cv")


def main(paths: list[str]) -> int:
    """Jalankan rencana untuk setiap PDF; kembalikan exit code (1 jika ada yang gagal)."""
    files = [Path(p).expanduser() for p in paths] or sorted(DEFAULT_DIR.glob("*.pdf"))
    if not files:
        print("Tidak ada PDF. Pemakaian: make sample-plan ARGS=path/ke/cv.pdf")
        return 2
    from src.config.database import VECTOR_DB  # import di sini: test tidak butuh koneksi Qdrant
    from src.config.llm import EMBEDDING, LLM_MODEL

    failed = outside_total = skills_total = 0
    for path in files:
        ingest = ingest_cv(path.read_bytes(), llm=LLM_MODEL)
        if ingest.profile is None:
            print(f"{path.name}: profil GAGAL ({ingest.reason})")
            failed += 1
            continue
        matches = match_jobs(
            ingest.profile, embeddings=EMBEDDING, vector_store=VECTOR_DB, db_path=DB_PATH
        )
        if matches.error or not matches.matches:
            print(f"{path.name}: pencocokan GAGAL")
            failed += 1
            continue
        gap = compute_skill_gap(matches.matches)
        started = time.perf_counter()
        result = generate_learning_plan(ingest.profile, gap, llm=LLM_MODEL)
        elapsed = time.perf_counter() - started
        if result.plan is None:
            print(f"{path.name}: rencana GAGAL (gap={len(gap.missing)})")
            failed += 1
            continue

        gap_names = {skill.name for skill in gap.missing}
        gap_folded = {name.casefold() for name in gap_names}
        planned = [skill for week in result.plan.weeks for skill in week.skills]
        outside = [skill for skill in planned if skill.casefold() not in gap_folded]
        not_exact = [skill for skill in planned if skill not in gap_names]
        covered = len(gap_names & set(planned))
        outside_total += len(outside)
        skills_total += len(planned)
        print(
            f"{path.name}: lowongan={gap.jobs} gap={len(gap.missing)} "
            f"minggu={len(result.plan.weeks)}/{PLAN_WEEKS} "
            f"skill_di_rencana={len(planned)} gap_tercakup={covered}/{len(gap_names)} "
            f"di_luar_gap={len(outside)} beda_huruf={len(not_exact) - len(outside)} "
            f"minggu_tanpa_tugas={sum(1 for week in result.plan.weeks if not week.tasks)} "
            f"token={result.usage.input_tokens}+{result.usage.output_tokens} "
            f"waktu={elapsed:.1f} dtk"
        )
    print(f"\nTOTAL di_luar_gap={outside_total}/{skills_total} skill, gagal={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
