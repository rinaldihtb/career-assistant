"""Ukur surat lamaran dengan API asli.

    make sample-letter
    make sample-letter ARGS=~/Downloads/CV.pdf

Tanpa ARGS: 5 CV fiktif di tests/eval/cv/. Per CV: ``ingest_cv`` (1 LLM) + ``match_jobs``
(1 embedding), lalu surat untuk lowongan teratas dengan 3 pengaturan (``SETTINGS``), jadi
4 panggilan LLM per CV.

Yang dicetak hanya angka, nama file, ID, dan judul lowongan; teks CV, nama, skill, dan isi surat
tidak dicetak, dan tidak ada yang ditulis ke disk. Metrik per surat:
- ``skill_dikarang``: skill kosakata di surat yang tidak ada di profil (aturan 3 prompt);
  ``dari_lowongan``: bagian darinya yang disebut lowongan (skill kurang yang diklaim).
- ``angka_dikarang``: angka di surat yang tidak ada di input prompt (aturan 2).
- ``salam_ganda``: paragraf isi yang memuat salam pembuka atau penutup (aturan 5); salam
  ditambahkan Python, jadi versi LLM akan tampil dua kali.
- ``anonim``: surat menyebut "Pengiklan Anonim" sebagai perusahaan.
- ``kata`` dan selisihnya dari target, jumlah jabatan dan skill profil yang disebut, token,
  dan waktu.
Kode penyaring hanya ditambahkan jika salah satu metrik pelanggaran terukur lebih dari 0.
"""

import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.cv.ingest import ingest_cv
from src.cv.letter import (
    ANONYMOUS_COMPANIES,
    LetterSettings,
    generate_cover_letter,
    hiring_company,
    letter_prompt_input,
)
from src.cv.match import match_jobs
from src.cv.skills import build_cv_skills, vocab_in_text
from src.repositories.job_repository import get_jobs_by_ids

load_dotenv()
DB_PATH = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")
DEFAULT_DIR = Path("tests/eval/cv")
SETTINGS = (
    LetterSettings(language="id", words=250),
    LetterSettings(language="id", words=150),
    LetterSettings(language="en", words=350),
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_GREETING = re.compile(r"^\s*(yth\b|dear\b|kepada\b|dengan hormat)", re.IGNORECASE)
_CLOSING = re.compile(
    r"hormat saya|salam hormat|sincerely|best regards|kind regards", re.IGNORECASE
)


def main(paths: list[str]) -> int:
    """Buat dan ukur surat untuk setiap PDF; kembalikan exit code (1 jika ada yang gagal)."""
    files = [Path(p).expanduser() for p in paths] or sorted(DEFAULT_DIR.glob("*.pdf"))
    if not files:
        print("Tidak ada PDF. Pemakaian: make sample-letter ARGS=path/ke/cv.pdf")
        return 2
    from src.config.database import VECTOR_DB  # import di sini: test tidak butuh koneksi Qdrant
    from src.config.llm import EMBEDDING, LLM_MODEL

    totals = {
        "surat": 0,
        "gagal": 0,
        "skill": 0,
        "dari_lowongan": 0,
        "angka": 0,
        "salam": 0,
        "anonim": 0,
    }
    deviations: list[float] = []
    for path in files:
        ingest = ingest_cv(path.read_bytes(), llm=LLM_MODEL)
        if ingest.profile is None:
            print(f"{path.name}: profil GAGAL ({ingest.reason})")
            totals["gagal"] += 1
            continue
        matches = match_jobs(
            ingest.profile, embeddings=EMBEDDING, vector_store=VECTOR_DB, db_path=DB_PATH
        )
        if matches.error or not matches.matches:
            print(f"{path.name}: pencocokan GAGAL")
            totals["gagal"] += 1
            continue
        match = matches.matches[0]
        jobs = get_jobs_by_ids(DB_PATH, [match.job_id])
        description = jobs[0].get("job_description") if jobs else None
        prompt_text = "\n".join(letter_prompt_input(ingest.profile, match, description))
        profile_vocab = build_cv_skills(ingest.profile.skills).vocab
        job_vocab = vocab_in_text(description) | set(match.missing_skills)
        titles = {exp.title.casefold() for exp in ingest.profile.experiences if exp.title}

        for settings in SETTINGS:
            job = f'job={match.job_id} "{match.job_title}"'
            label = f"{path.name} {job} {settings.language}/{settings.words}"
            started = time.perf_counter()
            result = generate_cover_letter(
                ingest.profile, match, description, llm=LLM_MODEL, settings=settings
            )
            elapsed = time.perf_counter() - started
            if result.letter is None:
                print(f"{label}: GAGAL")
                totals["gagal"] += 1
                continue

            paragraphs = result.letter.paragraphs
            body = "\n".join(paragraphs)
            folded = body.casefold()
            words = len(body.split())
            deviation = (words - settings.words) / settings.words
            invented_skills = vocab_in_text(body) - profile_vocab
            from_job = len(invented_skills & job_vocab)
            invented_numbers = set(_NUMBER.findall(body)) - set(_NUMBER.findall(prompt_text))
            double_greeting = sum(
                1 for p in paragraphs if _GREETING.search(p) or _CLOSING.search(p)
            )
            anonymous = sum(1 for name in ANONYMOUS_COMPANIES if name in folded)
            company = hiring_company(match.company_name)
            company_cell = "-" if company is None else str(company.casefold() in folded).lower()

            totals["surat"] += 1
            totals["skill"] += len(invented_skills)
            totals["dari_lowongan"] += from_job
            totals["angka"] += len(invented_numbers)
            totals["salam"] += double_greeting
            totals["anonim"] += anonymous
            deviations.append(deviation)
            print(
                f"{label}: kata={words} ({deviation:+.0%}) paragraf={len(paragraphs)} "
                f"skill_dikarang={len(invented_skills)} dari_lowongan={from_job} "
                f"angka_dikarang={len(invented_numbers)} "
                f"salam_ganda={double_greeting} anonim={anonymous} perusahaan={company_cell} "
                f"jabatan_disebut={sum(1 for t in titles if t in folded)}/{len(titles)} "
                f"skill_profil_disebut={len(vocab_in_text(body) & profile_vocab)} "
                f"token={result.usage.input_tokens}+{result.usage.output_tokens} "
                f"waktu={elapsed:.1f} dtk"
            )

    mean = sum(abs(d) for d in deviations) / len(deviations) if deviations else 0.0
    worst = max((abs(d) for d in deviations), default=0.0)
    print(
        f"\nTOTAL surat={totals['surat']} skill_dikarang={totals['skill']} "
        f"dari_lowongan={totals['dari_lowongan']} "
        f"angka_dikarang={totals['angka']} salam_ganda={totals['salam']} "
        f"anonim={totals['anonim']} selisih_kata rata-rata={mean:.0%} maks={worst:.0%} "
        f"gagal={totals['gagal']}"
    )
    return 1 if totals["gagal"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
