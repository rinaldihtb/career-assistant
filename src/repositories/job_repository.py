"""Read-only access to job postings for application and batch workflows."""

import os
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from qdrant_client import models

from src.models.jobs import Job, JobLocation

# Columns exposed to the batch/UI helpers.  Seed timestamps are deliberately
# omitted because they are not job-posting dates.
JOB_DETAIL_COLUMNS = (
    "id",
    "job_title",
    "company_name",
    "location",
    "work_type",
    "job_description",
    "salary_min",
    "salary_max",
    "salary_period",
    "salary_currency",
    "status",
)

DEFAULT_DB_PATH = "./data/database.sqlite"


def sqlite_path() -> str:
    """Path SQLite dari env ``DB_SQLITE_PATH``, default ``./data/database.sqlite``.

    Env dibaca saat dipanggil, bukan saat import, supaya ``load_dotenv`` dan
    ``monkeypatch.setenv`` di test berlaku. Dipakai fungsi yang menerima ``db_path``: fitur
    CV, gold SQL eval, ``match_role``, dan skrip sample.
    """
    return os.getenv("DB_SQLITE_PATH", DEFAULT_DB_PATH)


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open an SQLite database in read-only mode.

    ``mode=ro`` makes SQLite reject write operations at the file level.  An
    explicit existence check prevents SQLite from silently creating a blank
    database for a misspelled path.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database tidak ditemukan: {path}")
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def get_jobs_by_ids(db_path: str | Path, job_ids: Iterable[int]) -> list[dict[str, Any]]:
    """Return job details by ID, preserving first-occurrence input order."""
    ids = list(dict.fromkeys(int(job_id) for job_id in job_ids))
    if not ids:
        return []

    placeholders = ",".join("?" * len(ids))
    query = f"SELECT {', '.join(JOB_DETAIL_COLUMNS)} FROM jobs WHERE id IN ({placeholders})"
    with closing(connect_readonly(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, ids).fetchall()

    jobs_by_id = {row["id"]: dict(row) for row in rows}
    return [jobs_by_id[job_id] for job_id in ids if job_id in jobs_by_id]


class JobRepository:
    """Akses read-only data lowongan melalui ``SQLDatabase`` yang diinjeksi.

    Contoh inisialisasi pada composition root aplikasi::

        repository = JobRepository(db=SQL_DB, vector_db=VECTOR_DB)
    """

    def __init__(self, db: SQLDatabase, vector_db: VectorStore) -> None:
        self.db = db
        self.vector_db = vector_db

    def get_jobs_by_ids(self, job_ids: Iterable[int]) -> list[dict[str, Any]]:
        """Ambil lowongan berdasarkan ID dan pertahankan urutan input."""
        ids = list(dict.fromkeys(map(int, job_ids)))
        if not ids:
            return []

        parameters = {f"{index}": job_id for index, job_id in enumerate(ids)}
        placeholders = ", ".join(f":{name}" for name in parameters)
        columns = ", ".join(Job.model_fields)

        # Contoh jika ids = [10, 25]:
        # parameters   -> {"id_0": 10, "id_1": 25}
        # placeholders -> ":id_0, :id_1"
        # query        -> SELECT id, ... FROM jobs WHERE id IN (:id_0, :id_1)
        rows = self.db._execute(
            f"SELECT {columns} FROM jobs WHERE id IN ({placeholders})", parameters=parameters
        )

        jobs_by_id = {job.id: job.model_dump() for job in map(Job.model_validate, rows)}

        # Hasil: [{"id": 10, "job_title": "Backend Engineer", ...}, ...]
        # Urutannya mengikuti ``ids``: [10, 25].
        return [jobs_by_id[job_id] for job_id in ids if job_id in jobs_by_id]

    def get_job_locations(self) -> list[tuple[int, str]]:
        """Ambil pasangan ID dan lokasi untuk seluruh lowongan yang memiliki lokasi."""
        columns = ", ".join(JobLocation.model_fields)
        rows = self.db._execute(
            f"SELECT {columns} FROM jobs WHERE location IS NOT NULL AND location != ''"
        )
        locations: list[tuple[int, str]] = []

        for row in rows:
            location = JobLocation.model_validate(row)
            locations.append((location.id, location.location))

        return locations

    def get_job_titles(self) -> list[str]:
        """Ambil judul lowongan yang berbeda, untuk mengenali pesan berisi judul saja."""
        rows = self.db._execute("SELECT DISTINCT job_title FROM jobs WHERE job_title IS NOT NULL")
        return [row["job_title"] for row in rows]

    def get_job_listings(self) -> list[dict[str, Any]]:
        """Kolom daftar dan deskripsi semua lowongan, urut ID, untuk halaman Cari Lowongan.

        ``job_description`` NULL dikembalikan sebagai string kosong.
        """
        rows = self.db._execute(
            "SELECT id, job_title, company_name, location, work_type, salary_min, salary_max, "
            "COALESCE(job_description, '') AS job_description FROM jobs ORDER BY id"
        )
        return [dict(row) for row in rows]

    def execute_sql(self, sql: str, *, max_rows: int) -> list[dict[str, Any]]:
        """Jalankan query SQL agent pada koneksi SQLite yang hanya boleh membaca.

        ``PRAGMA query_only`` membuat SQLite menolak setiap penulisan di koneksi ini,
        termasuk ``WITH ... DELETE``, terlepas dari validasi di SQL agent. Query
        dikirim apa adanya ke driver (tanpa parsing parameter ``:nama``), dan hanya
        ``max_rows`` baris pertama yang diambil walaupun LIMIT di query lebih besar.

        Raises:
            sqlalchemy.exc.DBAPIError: Query ditolak atau gagal di SQLite. Pesannya
                dipakai SQL agent sebagai umpan balik saat retry.
        """
        with self.db._engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA query_only = ON")
            try:
                result = connection.exec_driver_sql(sql)
                return [dict(row._mapping) for row in result.fetchmany(max_rows)]
            finally:
                connection.rollback()
                connection.exec_driver_sql("PRAGMA query_only = OFF")

    def get_jobs_schema(self) -> str:
        """Ambil DDL tabel ``jobs`` dari database untuk prompt SQL agent."""
        return self.db.get_table_info(table_names=["jobs"])

    def similarity_search_with_score(
        self, query: str, k: int, filter: models.Filter | None = None
    ) -> list[tuple[Document, float]]:
        return self.vector_db.similarity_search_with_score(query, k, filter)


def get_job_descriptions(db_path: str | Path) -> list[tuple[int, str]]:
    """Ambil pasangan ``(id, job_description)`` semua lowongan.

    Deskripsi NULL dikembalikan sebagai string kosong agar pemanggil dapat
    langsung memprosesnya saat membangun indeks skill.
    """
    with closing(connect_readonly(db_path)) as conn:
        return conn.execute("SELECT id, COALESCE(job_description, '') FROM jobs").fetchall()


def get_job_locations(db_path: str | Path) -> list[tuple[int, str]]:
    """Return ``(id, location)`` pairs for every job with a non-empty location."""
    with closing(connect_readonly(db_path)) as conn:
        return conn.execute(
            "SELECT id, location FROM jobs WHERE location IS NOT NULL AND location != ''"
        ).fetchall()
