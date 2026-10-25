"""Uji manual ekstraksi profil CV dengan LLM asli.

    make sample-cv ARGS=~/Downloads/CV.pdf

Mencetak profil, lama pengalaman hasil hitungan Python, panjang teks, token, dan waktu.
Isi CV hanya ditampilkan di terminal ini; tidak ada yang ditulis ke disk. Biaya:
1 panggilan LLM per file.
"""

import sys
import time
from pathlib import Path

from src.cv.pdf_text import CVRejectedError, extract_cv_text
from src.cv.profile import extract_cv_profile


def main(paths: list[str]) -> int:
    """Jalankan ekstraksi untuk setiap path PDF; kembalikan exit code."""
    if not paths:
        print("Pemakaian: make sample-cv ARGS=path/ke/cv.pdf")
        return 2
    from src.config.llm import LLM_MODEL

    for path in paths:
        print(f"\n=== {path}")
        try:
            cv_text = extract_cv_text(Path(path).expanduser().read_bytes())
        except CVRejectedError as error:
            print(f"DITOLAK ({error.reason}): {error}")
            continue
        started = time.perf_counter()
        result = extract_cv_profile(cv_text.text, llm=LLM_MODEL)
        elapsed = time.perf_counter() - started
        print(f"halaman={cv_text.pages} karakter={len(cv_text.text)} terpotong={cv_text.truncated}")
        if result.profile is None:
            print(f"GAGAL: {result.error}")
        else:
            print(result.profile.model_dump_json(indent=2))
            print(f"pengalaman: {result.profile.experience_label}")
        print(
            f"llm_calls={result.usage.llm_calls} input={result.usage.input_tokens} "
            f"output={result.usage.output_tokens} waktu={elapsed:.2f} dtk"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
