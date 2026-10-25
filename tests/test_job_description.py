"""Perapian deskripsi lowongan menjadi bagian baku (tanpa Streamlit dan tanpa LLM)."""

import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import pytest

from src.components.job_description import (
    clean_lines,
    description_to_markdown,
    escape_markdown,
    header_key,
    structure_description,
)

REAL_DB = Path(__file__).resolve().parents[1] / "data" / "database.sqlite"


def _sections(text):
    return {section.key: section.items for section in structure_description(text)}


# --- Judul bagian ------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ("Kualifikasi", "kualifikasi"),
        ("Requirements:", "kualifikasi"),
        ("What we're looking for", "kualifikasi"),
        ("Soft Skills", "kualifikasi"),
        ("Persyaratan untuk peran ini", "kualifikasi"),
        ("Key Responsibilities", "tanggung_jawab"),
        ("Main Job Description", "tanggung_jawab"),
        ("Tugas & Tanggung Jawab", "tanggung_jawab"),
        ("What You\u2019ll Do", "tanggung_jawab"),
        ("What we offer", "benefit"),
        ("What you will get", "benefit"),
        ("Why Join Us?", "benefit"),
        ("Benefit(s):", "benefit"),
        ("Keuntungan Bergabung dengan Kami", "benefit"),
        ("Fasilitas:", "benefit"),
        ("About the role", "ringkasan"),
        ("Tentang Perusahaan", "ringkasan"),
        ("Sistem Kerja", "info"),
        ("Experience", "kualifikasi"),
    ],
)
def test_header_variants_map_to_standard_sections(line, key):
    assert header_key(line) == key


@pytest.mark.parametrize(
    "line",
    [
        "Experience in SQL",  # kata umum hanya judul jika berdiri sendiri
        "Notebook Ownership Program.",  # "note" harus kata utuh
        "Tunjangan Kesehatan & BPJS Ketenagakerjaan",  # kata kunci umum di item panjang
        "Mampu bekerja dalam tim dan memiliki kemampuan komunikasi yang baik",
    ],
)
def test_ordinary_lines_are_not_headers(line):
    assert header_key(line) is None


def test_sentence_fragment_is_not_header():
    lines = clean_lines("We are looking for a\nperson who loves data")
    assert lines == [("text", "We are looking for a person who loves data")]


def test_intro_sentence_completed_by_merge_becomes_header():
    lines = clean_lines("We are looking for a\ncandidate who has:\n- SQL")
    assert lines == [
        ("header:kualifikasi", "We are looking for a candidate who has"),
        ("bullet", "SQL"),
    ]


# --- Pembersihan baris -------------------------------------------------------

RAW = (
    "KUALIFIKASI\xa0\xa0 :\n"
    "-\xa0\xa0\xa0 Pendidikan minimal S1\n"
    "•\n"
    "Proficient in\n"
    "Python untuk otomasi\n"
    "We are seeking\n"
    "top-tier graduates (\n"
    "GPA \u2265 3.5/4.0\n"
    ").\n"
    "\U0001f539 Analisis data\u2060 penjualan $10"
)


def test_clean_lines_rules():
    assert clean_lines(RAW) == [
        ("header:kualifikasi", "KUALIFIKASI"),
        ("bullet", "Pendidikan minimal S1"),
        ("bullet", "Proficient in Python untuk otomasi"),
        ("text", "We are seeking top-tier graduates (GPA \u2265 3.5/4.0)."),
        ("text", "Analisis data penjualan $10"),
    ]


def test_leading_comma_line_continues_sentence():
    lines = clean_lines("Lead the analytics strategy\n, ensuring alignment with goals.")
    assert lines == [("text", "Lead the analytics strategy, ensuring alignment with goals.")]


# --- Struktur bagian ---------------------------------------------------------


def test_sections_follow_standard_order_and_merge_duplicates():
    text = (
        "Kami perusahaan teknologi.\n"
        "Benefit\n- BPJS\n"
        "Requirements\n- SQL\n"
        "Responsibilities\n- Membuat dashboard\n"
        "Soft Skills\n- Komunikatif\n"
    )
    sections = structure_description(text)

    assert [section.title for section in sections] == [
        "Ringkasan",
        "Tanggung Jawab",
        "Kualifikasi",
        "Benefit",
    ]
    assert _sections(text)["kualifikasi"] == [("bullet", "SQL"), ("bullet", "Komunikatif")]


def test_description_without_headers_is_one_section():
    sections = structure_description("Lead the team.\nBuild dashboards.")
    assert [(section.title, len(section.items)) for section in sections] == [("Deskripsi", 2)]


def test_long_original_header_kept_as_intro():
    text = "Tentang PT Cielco Ivin Anugerah Perkasa:\nDistributor alat teknik."
    assert _sections(text)["ringkasan"][0] == ("intro", "Tentang PT Cielco Ivin Anugerah Perkasa")


def test_key_value_line_goes_to_info():
    assert _sections("Kualifikasi\n- SQL\nLocation: Jakarta (Hybrid)")["info"] == [
        ("bullet", "Location: Jakarta (Hybrid)")
    ]


# --- Markdown ----------------------------------------------------------------


def test_list_sections_are_numbered():
    markdown = description_to_markdown("Kualifikasi\n- S1 Statistik\n- Menguasai SQL")
    assert markdown == "**Kualifikasi**\n\n1. S1 Statistik\n2. Menguasai SQL"


def test_sub_labels_restart_numbering():
    text = (
        "Tanggung Jawab\n"
        "\U0001f539 Recruitment\n• Screening CV\n• Interview\n"
        "\U0001f539 Payroll\n• Hitung gaji\n"
    )
    assert description_to_markdown(text) == (
        "**Tanggung Jawab**\n\n*Recruitment*\n\n1. Screening CV\n2. Interview"
        "\n\n*Payroll*\n\n1. Hitung gaji"
    )


def test_summary_uses_paragraphs_and_bullet_runs():
    paragraph = (
        "Kami adalah perusahaan teknologi yang berkembang pesat di Indonesia. "
        "Kami membangun produk data untuk ribuan pelanggan di berbagai industri."
    )
    text = f"About us\n{paragraph}\nInovatif\nKolaboratif\nKualifikasi\n- SQL"
    assert description_to_markdown(text) == (
        f"**Ringkasan**\n\n{paragraph}\n\n- Inovatif\n- Kolaboratif\n\n**Kualifikasi**\n\n1. SQL"
    )


def test_escape_markdown():
    assert escape_markdown("C# *senior* [remote] $5") == "C\\# \\*senior\\* \\[remote\\] \\$5"


# --- Seluruh data asli -------------------------------------------------------


def _words(text):
    """Kata asli tanpa penomoran lama ("1", "a") yang memang diganti penomoran baru."""
    return Counter(
        w for w in re.findall(r"\w+", text.lower()) if not re.fullmatch(r"\d{1,2}|[a-z]", w)
    )


@pytest.mark.skipif(not REAL_DB.is_file(), reason="data/database.sqlite tidak tersedia")
def test_real_descriptions_lose_no_words_and_are_mostly_structured():
    """Jaminan utama: tidak ada kata yang hilang, dan sebagian besar punya bagian daftar."""
    with closing(sqlite3.connect(f"{REAL_DB.as_uri()}?mode=ro", uri=True)) as db:
        descriptions = [row[0] or "" for row in db.execute("SELECT job_description FROM jobs")]

    structured = 0
    for text in descriptions:
        kept = sum((_words(line) for _, line in clean_lines(text)), Counter())
        assert not _words(text) - kept
        keys = {section.key for section in structure_description(text)}
        structured += bool(keys & {"tanggung_jawab", "kualifikasi"})
    assert structured / len(descriptions) >= 0.9
