"""Pencarian dan filter halaman Cari Lowongan: parsing lokasi, kata kunci, urutan, filter."""

from pathlib import Path

import pytest

from repositories.job_search import (
    MAX_KEYWORD_TERMS,
    JobFilters,
    JobListing,
    filter_options,
    keyword_terms,
    load_listings,
    parse_location,
    province_label,
    search_jobs,
)

REAL_DB = Path(__file__).resolve().parents[1] / "data" / "database.sqlite"


def listing(job_id: int, title: str, text: str = "", company: str | None = None) -> JobListing:
    searchable = "\n".join((title, company or "", text)).casefold()
    return JobListing(
        id=job_id,
        job_title=title,
        company_name=company,
        location=None,
        work_type=None,
        salary_min=None,
        salary_max=None,
        search_text=searchable,
    )


def ids(jobs: list[JobListing]) -> list[int]:
    return [job.id for job in jobs]


# --- Lokasi -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("Jakarta Selatan, Jakarta Raya", ("Jakarta Selatan", "Jakarta Raya", None)),
        ("Jakarta Selatan, Jakarta Raya\n(Hibrid)", ("Jakarta Selatan", "Jakarta Raya", "Hibrid")),
        ("Jakarta Raya\n(Jarak jauh)", (None, "Jakarta Raya", "Jarak jauh")),
        ("Bali", (None, "Bali", None)),
        ("  Batam ,  Kepulauan  Riau ", ("Batam", "Kepulauan Riau", None)),
        ("", (None, None, None)),
        (None, (None, None, None)),
    ],
)
def test_parse_location(location, expected):
    assert parse_location(location) == expected


def test_province_label_uses_common_name():
    assert province_label("Jakarta Raya") == "DKI Jakarta"
    assert province_label("Jawa Timur") == "Jawa Timur"


def test_load_listings_parses_every_row(jobs_db):
    listings = load_listings(str(jobs_db))

    assert ids(listings) == [1, 2, 3, 4, 5, 6]
    hybrid = listings[3]
    assert (hybrid.city, hybrid.province, hybrid.mode) == (None, "Jakarta Raya", "Hibrid")
    assert hybrid.place == "DKI Jakarta"
    assert listings[0].place == "Jakarta Selatan"


def test_filter_options_counts_and_orders_by_count(jobs_db):
    options = filter_options(load_listings(str(jobs_db)))

    assert options.provinces[0] == ("Jakarta Raya", 3)
    assert dict(options.work_types) == {"Full time": 4, "Paruh waktu": 1, "Kontrak/Temporer": 1}
    assert options.modes == [("Hibrid", 1)]


# --- Kata kunci ---------------------------------------------------------------


def test_keyword_terms_are_lowercase_unique_and_capped():
    assert keyword_terms("  Data  ANALYST data ") == ["data", "analyst"]
    assert len(keyword_terms(" ".join(f"k{i}" for i in range(20)))) == MAX_KEYWORD_TERMS


def test_keyword_searches_all_fields_and_requires_every_term(jobs_db):
    listings = load_listings(str(jobs_db))

    assert ids(search_jobs(listings, JobFilters(keyword="python"))) == [1, 5]
    assert ids(search_jobs(listings, JobFilters(keyword="data surabaya"))) == [2]
    assert ids(search_jobs(listings, JobFilters(keyword="PT Gamma"))) == [3]
    assert search_jobs(listings, JobFilters(keyword="python bandung")) == []


def test_short_terms_match_whole_words_only():
    listings = [listing(1, "AI Engineer"), listing(2, "Admin", "Detail oriented, maintain data")]

    assert ids(search_jobs(listings, JobFilters(keyword="ai"))) == [1]


def test_longer_terms_match_word_start():
    listings = [listing(1, "Staf", "administrasi gudang"), listing(2, "Staf", "sysadmin")]

    assert ids(search_jobs(listings, JobFilters(keyword="admin"))) == [1]


def test_title_phrase_ranks_before_title_words_company_and_description():
    listings = [
        listing(1, "Staf Gudang", "dibutuhkan data analyst"),
        listing(2, "Analyst Data Senior", "data"),
        listing(3, "Data Analyst"),
        listing(4, "Staf", "analyst", company="Data Prima"),
    ]

    assert ids(search_jobs(listings, JobFilters(keyword="data analyst"))) == [3, 2, 4, 1]


def test_without_keyword_keeps_id_order(jobs_db):
    listings = load_listings(str(jobs_db))

    assert ids(search_jobs(listings, JobFilters())) == [1, 2, 3, 4, 5, 6]


# --- Filter -------------------------------------------------------------------


def test_filters_province_work_type_and_mode(jobs_db):
    listings = load_listings(str(jobs_db))

    assert ids(search_jobs(listings, JobFilters(province="Jakarta Raya"))) == [1, 4, 6]
    assert ids(search_jobs(listings, JobFilters(work_type="Paruh waktu"))) == [2]
    assert ids(search_jobs(listings, JobFilters(mode="Hibrid"))) == [4]
    combined = JobFilters(keyword="data", province="Jakarta Raya", work_type="Full time")
    assert ids(search_jobs(listings, combined)) == [1]


def test_min_salary_uses_upper_bound_and_drops_jobs_without_salary(jobs_db):
    listings = load_listings(str(jobs_db))

    assert ids(search_jobs(listings, JobFilters(min_salary=10_000_000))) == [1, 4]
    assert ids(search_jobs(listings, JobFilters(min_salary=7_000_000))) == [1, 4, 6]


# --- Database asli ------------------------------------------------------------


@pytest.mark.skipif(not REAL_DB.is_file(), reason="data/database.sqlite belum di-seed")
def test_real_database_every_location_has_a_province():
    listings = load_listings(str(REAL_DB))
    options = filter_options(listings)

    assert sum(count for _, count in options.provinces) == len(listings)
    assert {mode for mode, _ in options.modes} <= {"Hibrid", "Jarak jauh"}
    top = search_jobs(listings, JobFilters(keyword="data analyst"))[0]
    assert "data analyst" in top.job_title.casefold()
