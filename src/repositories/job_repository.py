"""Akses read-only ke database lowongan (SQLite). Tidak ada LLM di file ini."""

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

# Kolom yang dikirim ke agent/UI. created_at dan updated_at sengaja tidak ikut:
# isinya waktu seeding, bukan tanggal posting, jadi menyesatkan jika ditampilkan.
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


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Buka koneksi SQLite dalam mode read-only.

    ``mode=ro`` membuat SQLite menolak semua operasi tulis (INSERT, UPDATE,
    DELETE, DROP, CREATE) di level file, apa pun isi query-nya.

    Args:
        db_path: Path ke file SQLite, relatif terhadap working directory.

    Returns:
        Koneksi siap pakai. Pemanggil wajib menutupnya (pakai
        ``contextlib.closing``).

    Raises:
        FileNotFoundError: Jika file tidak ada. Tanpa cek ini, SQLite akan
            diam-diam membuat database kosong baru.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database tidak ditemukan: {path}")
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def get_jobs_by_ids(db_path: str | Path, job_ids: Iterable[int]) -> list[dict[str, Any]]:
    """Ambil detail lowongan berdasarkan daftar ID, urutan mengikuti input.

    Urutan input dipertahankan karena RAG agent mengirim ID yang sudah
    diurutkan dari skor tertinggi; ``WHERE id IN (...)`` di SQLite tidak
    menjamin urutan apa pun.

    Args:
        db_path: Path ke file SQLite.
        job_ids: ID lowongan. Duplikat dibuang, kemunculan pertama dipakai.

    Returns:
        Satu dict per lowongan dengan kolom ``JOB_DETAIL_COLUMNS``. ID yang
        tidak ada di database dilewati (misalnya vektor Qdrant basi setelah
        SQLite di-seed ulang). Input kosong langsung mengembalikan ``[]``
        tanpa membuka database.

    Raises:
        FileNotFoundError: Jika file database tidak ada.
        sqlite3.Error: Jika query gagal.
    """
    ids = list(dict.fromkeys(int(job_id) for job_id in job_ids))
    if not ids:
        return []

    # Yang disisipkan ke string hanya nama kolom konstanta dan tanda "?";
    # nilai ID tetap dikirim sebagai parameter.
    placeholders = ",".join("?" * len(ids))
    query = f"SELECT {', '.join(JOB_DETAIL_COLUMNS)} FROM jobs WHERE id IN ({placeholders})"
    with closing(connect_readonly(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, ids).fetchall()

    by_id = {row["id"]: dict(row) for row in rows}
    return [by_id[job_id] for job_id in ids if job_id in by_id]


def get_job_locations(db_path: str | Path) -> list[tuple[int, str]]:
    """Ambil pasangan ``(id, location)`` semua lowongan yang punya lokasi.

    Dipakai RAG agent untuk membangun kamus lokasi tanpa LLM. Dengan 473
    baris, mengambil semuanya lalu mencocokkan di Python lebih sederhana
    daripada menyusun query LIKE dinamis.

    Raises:
        FileNotFoundError: Jika file database tidak ada.
        sqlite3.Error: Jika query gagal.
    """
    with closing(connect_readonly(db_path)) as conn:
        return conn.execute(
            "SELECT id, location FROM jobs WHERE location IS NOT NULL AND location != ''"
        ).fetchall()
