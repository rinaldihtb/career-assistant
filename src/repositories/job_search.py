"""Pencarian dan filter lowongan untuk halaman Cari Lowongan, tanpa LLM dan tanpa embedding.

Semua lowongan dibaca sekali per proses (``load_listings``), lalu kata kunci dan filter
dicocokkan di Python. Dengan 473 baris, cara ini lebih sederhana dan lebih mudah dites daripada
menyusun query LIKE dinamis (pola sama dengan ``job_repository.get_job_locations``).

Lokasi mentah (93 nilai) dipecah dengan aturan tetap: baris pertama "Kota, Provinsi" atau hanya
"Provinsi"; baris kedua "(Hibrid)" atau "(Jarak jauh)" adalah mode kerja.
"""

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from functools import lru_cache

from repositories.job_repository import connect_readonly

# Nama provinsi di data -> label yang lazim dibaca pengguna. Nilai filter tetap nama di data.
PROVINCE_LABELS = {"Jakarta Raya": "DKI Jakarta", "West Sumatra": "Sumatera Barat"}
WORK_MODES = {"(hibrid)": "Hibrid", "(jarak jauh)": "Jarak jauh"}
MAX_KEYWORD_TERMS = 8
# Kata kunci sependek ini hanya cocok sebagai kata utuh. Sebagai potongan kata, "ai" cocok
# dengan 436 dari 473 lowongan ("detail", "maintain") dan "it" dengan 455; sebagai kata utuh
# 31 dan 52. Kata yang lebih panjang cocok di awal kata, sehingga "admin" tetap menemukan
# "administrasi" (82 lowongan; 11 jika harus kata utuh).
WHOLE_WORD_MAX_CHARS = 3


@dataclass(frozen=True)
class JobListing:
    """Satu lowongan untuk daftar hasil.

    Attributes:
        province: Nama provinsi seperti di data (bagian terakhir baris pertama lokasi).
        mode: "Hibrid", "Jarak jauh", atau ``None`` jika lokasi tidak menyebutnya.
        search_text: Judul, perusahaan, lokasi, dan deskripsi dalam huruf kecil; hanya untuk
            pencocokan kata kunci.
    """

    id: int
    job_title: str
    company_name: str | None
    location: str | None
    work_type: str | None
    salary_min: int | None
    salary_max: int | None
    city: str | None = None
    province: str | None = None
    mode: str | None = None
    search_text: str = field(default="", repr=False)

    @property
    def place(self) -> str:
        """Lokasi satu baris untuk kartu: kota jika ada, selain itu label provinsi."""
        if self.city:
            return self.city
        return province_label(self.province) if self.province else ""


@dataclass(frozen=True)
class JobFilters:
    """Isian form pencarian. Nilai kosong/``None``/0 berarti filter itu tidak dipakai."""

    keyword: str = ""
    province: str | None = None
    work_type: str | None = None
    mode: str | None = None
    min_salary: int = 0


@dataclass(frozen=True)
class FilterOptions:
    """Pilihan dropdown beserta jumlah lowongannya, urut jumlah terbanyak."""

    provinces: list[tuple[str, int]]
    work_types: list[tuple[str, int]]
    modes: list[tuple[str, int]]


def parse_location(location: str | None) -> tuple[str | None, str | None, str | None]:
    """Pecah lokasi mentah menjadi ``(kota, provinsi, mode)``.

    "Jakarta Selatan, Jakarta Raya\\n(Hibrid)" -> ("Jakarta Selatan", "Jakarta Raya", "Hibrid");
    "Bali" -> (None, "Bali", None). Bagian yang tidak ada bernilai ``None``.
    """
    lines = [" ".join(line.split()) for line in (location or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None, None, None
    mode = next(
        (WORK_MODES[line.casefold()] for line in lines[1:] if line.casefold() in WORK_MODES), None
    )
    parts = [part.strip() for part in lines[0].split(",") if part.strip()]
    if not parts:
        return None, None, mode
    city = parts[0] if len(parts) > 1 else None
    return city, parts[-1], mode


def province_label(province: str) -> str:
    """Label provinsi untuk pengguna, misalnya "Jakarta Raya" -> "DKI Jakarta"."""
    return PROVINCE_LABELS.get(province, province)


@lru_cache(maxsize=4)
def load_listings(db_path: str) -> tuple[JobListing, ...]:
    """Baca semua lowongan sekali per proses, urut ID.

    Database read-only dan hanya berubah saat seeding ulang (aplikasi di-restart), jadi cache
    seumur proses aman (sama dengan ``cv.match.load_job_skill_index``).

    Raises:
        FileNotFoundError: Jika file database tidak ada.
        sqlite3.Error: Jika query gagal.
    """
    with closing(connect_readonly(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, job_title, company_name, location, work_type, salary_min, salary_max, "
            "COALESCE(job_description, '') FROM jobs ORDER BY id"
        ).fetchall()
    listings = []
    for job_id, title, company, location, work_type, salary_min, salary_max, text in rows:
        city, province, mode = parse_location(location)
        searchable = "\n".join(part or "" for part in (title, company, location, text))
        listings.append(
            JobListing(
                id=job_id,
                job_title=" ".join((title or "").split()),
                company_name=company,
                location=location,
                work_type=work_type,
                salary_min=salary_min,
                salary_max=salary_max,
                city=city,
                province=province,
                mode=mode,
                search_text=searchable.casefold(),
            )
        )
    return tuple(listings)


def filter_options(listings: Iterable[JobListing]) -> FilterOptions:
    """Hitung pilihan provinsi, tipe kerja, dan mode kerja dari data."""
    listings = list(listings)

    def counted(values: Iterable[str | None]) -> list[tuple[str, int]]:
        counts = Counter(value for value in values if value)
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    return FilterOptions(
        provinces=counted(job.province for job in listings),
        work_types=counted(job.work_type for job in listings),
        modes=counted(job.mode for job in listings),
    )


def keyword_terms(keyword: str) -> list[str]:
    """Kata kunci dalam huruf kecil, tanpa duplikat, maksimal ``MAX_KEYWORD_TERMS``."""
    return list(dict.fromkeys(keyword.casefold().split()))[:MAX_KEYWORD_TERMS]


def term_pattern(term: str) -> re.Pattern[str]:
    """Pola satu kata kunci: kata utuh untuk kata pendek, awal kata untuk yang lebih panjang."""
    end = r"(?!\w)" if len(term) <= WHOLE_WORD_MAX_CHARS else ""
    return re.compile(rf"(?<!\w){re.escape(term)}{end}")


def search_jobs(listings: Sequence[JobListing], filters: JobFilters) -> list[JobListing]:
    """Lowongan yang lolos semua filter dan memuat setiap kata kunci.

    Kata kunci dicari di judul, perusahaan, lokasi, dan deskripsi. Dengan kata kunci, hasil
    diurutkan: frasa utuh di judul, lalu jumlah kata kunci di judul, lalu di nama perusahaan,
    lalu ID. Tanpa kata kunci, urutan ID. Filter gaji hanya meloloskan lowongan yang
    mencantumkan gaji dengan batas atas (atau batas bawah jika hanya itu) minimal
    ``min_salary``.
    """
    terms = keyword_terms(filters.keyword)
    patterns = [term_pattern(term) for term in terms]
    results = [
        job
        for job in listings
        if _passes_filters(job, filters)
        and all(pattern.search(job.search_text) for pattern in patterns)
    ]
    if not patterns:
        return results
    phrase = term_pattern(" ".join(terms)) if len(terms) > 1 else None

    def relevance(job: JobListing) -> tuple[int, int, int, int]:
        title = job.job_title.casefold()
        company = (job.company_name or "").casefold()
        in_title = sum(1 for pattern in patterns if pattern.search(title))
        in_company = sum(1 for pattern in patterns if pattern.search(company))
        has_phrase = bool(phrase.search(title)) if phrase else in_title == len(patterns)
        return (0 if has_phrase else 1, -in_title, -in_company, job.id)

    return sorted(results, key=relevance)


def _passes_filters(job: JobListing, filters: JobFilters) -> bool:
    if filters.province and job.province != filters.province:
        return False
    if filters.work_type and job.work_type != filters.work_type:
        return False
    if filters.mode and job.mode != filters.mode:
        return False
    if filters.min_salary:
        top = job.salary_max or job.salary_min
        if not top or top < filters.min_salary:
            return False
    return True
