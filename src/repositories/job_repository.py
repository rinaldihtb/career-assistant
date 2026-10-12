"""Akses read-only ke database lowongan (SQLite). Tidak ada LLM di file ini."""

import sqlite3
from pathlib import Path


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
