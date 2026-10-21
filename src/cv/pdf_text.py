"""Ekstraksi teks CV dari PDF yang punya lapisan teks (tanpa OCR).

Kenapa membaca lapisan teks, bukan OCR: CV yang diekspor dari Word, Google Docs,
atau Canva hampir selalu punya lapisan teks. Membacanya butuh milidetik dan
hasilnya persis; OCR butuh binary sistem di container, 1-3 detik per halaman, dan
bisa salah huruf. PDF tanpa teks (hasil scan, foto, atau teks yang diubah jadi
gambar) ditolak dengan pesan yang menjelaskan cara memperbaikinya.

Kenapa pypdf: Python murni berlisensi BSD (tanpa binary), dan pada korpus kalibrasi
(CV 1 kolom, 2 kolom berbasis tabel, 2 kolom posisi bebas, 3 halaman) hasilnya
setara pypdfium2: semua skill terbaca dan kolom tidak tercampur. pypdfium2 lebih
cepat (1-7 ms vs 3-65 ms per CV) tapi selisih itu tidak terasa di samping satu
panggilan LLM. Mode ``extraction_mode="layout"`` sengaja tidak dipakai karena
justru menggabungkan baris dari dua kolom.

Fungsi di modul ini tidak memanggil LLM dan tidak menyimpan apa pun ke disk.
"""

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_PAGES = 5
# Rata-rata karakter non-spasi per halaman. CV berteks di korpus kalibrasi: 358-2.839;
# PDF hasil scan: 0. Ambang 100 memberi jarak aman ke dua sisi.
MIN_CHARS_PER_PAGE = 100
# Satu halaman CV padat sekitar 4.000 karakter, jadi 5 halaman sekitar 20.000. Batas ini
# hanya memotong kasus tidak wajar (misalnya teks tersembunyi berukuran kecil) agar
# token panggilan LLM berikutnya tetap terkendali.
MAX_TEXT_CHARS = 20_000


class CVRejectedError(Exception):
    """PDF tidak bisa dipakai sebagai CV.

    ``str(error)`` adalah pesan untuk pengguna (Bahasa Indonesia, siap ditampilkan
    dengan ``st.error``). ``reason`` adalah kode singkat untuk test dan logika UI:
    ``empty``, ``too_large``, ``not_pdf``, ``encrypted``, ``unreadable``,
    ``too_many_pages``, atau ``no_text``.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class CVText:
    """Teks CV hasil ekstraksi.

    Attributes:
        text: Teks yang sudah dirapikan, antarhalaman dipisah baris kosong.
        pages: Jumlah halaman PDF.
        truncated: True jika teks dipotong di ``MAX_TEXT_CHARS``.
    """

    text: str
    pages: int
    truncated: bool


def extract_cv_text(data: bytes) -> CVText:
    """Baca teks CV dari isi file PDF.

    Pemeriksaan murah dijalankan lebih dulu (ukuran, tanda tangan file) sebelum PDF
    di-parse, supaya file yang jelas salah tidak sempat diproses.

    Args:
        data: Isi file mentah, misalnya ``UploadedFile.getvalue()`` dari Streamlit.

    Returns:
        ``CVText`` berisi teks yang sudah dirapikan.

    Raises:
        CVRejectedError: Satu-satunya exception yang dilempar. Error internal pypdf
            (PDF rusak, enkripsi yang tidak didukung) dibungkus menjadi error ini.
    """
    if not data:
        raise CVRejectedError("empty", "File kosong. Pilih file PDF CV-mu.")
    if len(data) > MAX_FILE_BYTES:
        size_mb = len(data) / (1024 * 1024)
        raise CVRejectedError(
            "too_large",
            f"Ukuran file {size_mb:.1f} MB, maksimal 5 MB. Kompres atau ekspor ulang CV-mu.",
        )
    # Spesifikasi PDF mengizinkan header berada di 1.024 byte pertama.
    if b"%PDF-" not in data[:1024]:
        raise CVRejectedError(
            "not_pdf", "File ini bukan PDF. Ekspor CV-mu sebagai PDF lalu upload lagi."
        )

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # tipe error pypdf untuk file rusak beragam
        raise _unreadable() from exc

    if reader.is_encrypted:
        try:
            # Banyak PDF hanya dikunci untuk edit/cetak (owner password) dengan user
            # password kosong; isinya tetap bisa dibaca.
            decrypted = reader.decrypt("")
        except Exception:  # noqa: BLE001 - mis. enkripsi AES butuh paket cryptography
            decrypted = 0
        if not decrypted:
            raise CVRejectedError(
                "encrypted",
                "PDF ini dikunci atau memakai proteksi yang belum didukung. Simpan ulang "
                "tanpa password (misalnya lewat Print, Save as PDF), lalu upload lagi.",
            )

    try:
        page_count = len(reader.pages)
        if page_count > MAX_PAGES:
            raise CVRejectedError(
                "too_many_pages",
                f"PDF berisi {page_count} halaman, maksimal {MAX_PAGES}. Upload CV-mu saja, "
                "tanpa portofolio atau lampiran.",
            )
        pages = [_clean(page.extract_text() or "") for page in reader.pages]
    except CVRejectedError:
        raise
    except Exception as exc:  # error parsing halaman dari pypdf beragam
        raise _unreadable() from exc

    if page_count == 0:
        raise CVRejectedError("empty", "PDF tidak berisi halaman. Pilih file PDF CV-mu.")

    text = "\n\n".join(page for page in pages if page)
    visible_chars = len(re.sub(r"\s", "", text))
    if visible_chars / page_count < MIN_CHARS_PER_PAGE:
        raise CVRejectedError(
            "no_text",
            "PDF ini tidak punya teks yang bisa dibaca (kemungkinan hasil scan, foto, atau "
            "teks yang diubah menjadi gambar). Ekspor ulang CV dari Word, Google Docs, atau "
            "Canva sebagai PDF yang teksnya bisa diblok dan di-copy, lalu upload lagi.",
        )

    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS]
    return CVText(text=text, pages=page_count, truncated=truncated)


def _clean(text: str) -> str:
    """Rapikan spasi tanpa membuang struktur baris (judul bagian, butir daftar)."""
    text = text.replace("\x00", "")
    lines = [re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _unreadable() -> CVRejectedError:
    return CVRejectedError(
        "unreadable",
        "PDF tidak bisa dibaca, mungkin rusak. Ekspor ulang CV-mu sebagai PDF lalu upload lagi.",
    )
