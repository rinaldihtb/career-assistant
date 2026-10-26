"""Satu pintu masuk CV: PDF -> teks -> profil (1 panggilan LLM).

Semua tempat upload CV memakai fungsi ini, sehingga validasi, ekstraksi, dan pesan error
sama. Persetujuan pengguna dicek UI sebelum fungsi ini dipanggil.

Teks CV hanya hidup selama pemanggilan. Yang dikembalikan hanya profil (disimpan pemanggil
di ``st.session_state``) dan angka. Log tidak memuat teks CV maupun pesan error
LLM, karena pesan "output tidak valid" bisa berisi potongan output model, yaitu data CV.
"""

import logging
from datetime import date

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.cv.contacts import extract_contacts
from src.cv.pdf_text import CVRejectedError, extract_cv_text
from src.cv.profile import CVProfile, extract_cv_profile
from src.models.agent import LLMUsage

logger = logging.getLogger(__name__)

# Kunci st.session_state untuk CV aktif, dibaca semua halaman: {profile, version, matches, ...}.
CV_STATE_KEY = "cv"
PROFILE_FAILED = (
    "CV terbaca, tetapi profilnya gagal diekstrak. Coba lagi beberapa saat lagi. Jika tetap "
    "gagal, ekspor ulang CV dari Word atau Google Docs lalu upload lagi."
)


class IngestResult(BaseModel):
    """Hasil ``ingest_cv``. ``profile`` terisi jika berhasil; selain itu ``error``.

    Attributes:
        error: Pesan untuk pengguna (Bahasa Indonesia, siap untuk ``st.error``).
        reason: Kode singkat untuk test dan logika UI: alasan ``CVRejectedError``
            (``not_pdf``, ``no_text``, ...) atau ``profile`` jika LLM gagal.
        truncated: Teks CV dipotong di batas ``pdf_text.MAX_TEXT_CHARS``.
    """

    profile: CVProfile | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    pages: int = 0
    chars: int = 0
    truncated: bool = False
    error: str | None = None
    reason: str | None = None


def ingest_cv(data: bytes, *, llm: BaseChatModel, today: date | None = None) -> IngestResult:
    """Validasi PDF, baca teksnya, lalu ekstrak profil dengan tepat satu panggilan LLM.

    Kontak untuk kop surat (``CVProfile.contacts``) diambil dari teks yang sama dengan regex,
    tanpa LLM.

    PDF yang ditolak (bukan PDF, terlalu besar, hasil scan, dll.) berhenti sebelum LLM: 0
    panggilan, 0 biaya.

    Args:
        data: Isi file mentah, misalnya ``UploadedFile.getvalue()``.
        llm: Chat model untuk ``extract_cv_profile``.
        today: Tanggal acuan lama pengalaman; diisi di test.

    Returns:
        ``IngestResult``. Tidak pernah melempar exception.
    """
    try:
        cv = extract_cv_text(data)
    except CVRejectedError as error:
        return IngestResult(error=str(error), reason=error.reason)

    meta = {"pages": cv.pages, "chars": len(cv.text), "truncated": cv.truncated}
    result = extract_cv_profile(cv.text, llm=llm, today=today)
    if result.profile is None:
        logger.warning(
            "Ekstraksi profil CV gagal (%d halaman, %d karakter)", cv.pages, len(cv.text)
        )
        return IngestResult(usage=result.usage, error=PROFILE_FAILED, reason="profile", **meta)
    profile = result.profile.model_copy(update={"contacts": extract_contacts(cv.text)})
    return IngestResult(profile=profile, usage=result.usage, **meta)
