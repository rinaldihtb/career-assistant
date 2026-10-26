"""Kontak pelamar dari teks CV dengan regex, tanpa LLM: telepon, email, LinkedIn, GitHub.

Dipakai untuk kop surat lamaran. Kontak hanya disimpan di profil sesi (``CVProfile.contacts``)
dan tidak pernah masuk prompt LLM: prompt planner, rencana belajar, dan surat menyusun teks
profil dari field yang dipilih satu per satu, dan field ini tidak termasuk.

Regex, bukan field baru di skema LLM: format kontak baku, pencocokan teks gratis dan
deterministik, dan nomor telepon atau email tidak perlu dikirim ke LLM. Di 5 CV fiktif set eval
semua email dan nomor telepon terbaca dengan pola ini.
"""

import re

# Nomor Indonesia: +62 / 62 / (+62) / 0, lalu 8xx, dengan pemisah spasi, titik, atau tanda hubung.
_PHONE = re.compile(r"(?<![\w+])(?:\(?\+?62\)?[\s.-]?|0)8\d{1,3}(?:[\s.-]?\d{2,5}){1,3}(?!\d)")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_LINKEDIN = re.compile(r"(?:https?://)?(?:[\w-]+\.)?linkedin\.com/in/[\w%.-]+", re.IGNORECASE)
_GITHUB = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w-]+", re.IGNORECASE)
_URL_PREFIX = re.compile(r"^(?:https?://)?(?:www\.)?", re.IGNORECASE)
# Nomor Indonesia 10-13 digit termasuk awalan 0 (atau 11-14 dengan 62).
MIN_PHONE_DIGITS = 10
MAX_PHONE_DIGITS = 14


def extract_contacts(text: str) -> list[str]:
    """Kontak pertama per jenis, urut telepon, email, LinkedIn, GitHub.

    URL ditulis tanpa ``https://`` dan ``www.`` ("linkedin.com/in/nama"); telepon dan email
    ditulis seperti di CV. Jenis yang tidak ditemukan dilewati.
    """
    found = [_first_phone(text), _first(_EMAIL, text)]
    found += [_short_url(_first(pattern, text)) for pattern in (_LINKEDIN, _GITHUB)]
    return [item for item in found if item]


def _first(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(0).strip(" .,;") if match else ""


def _first_phone(text: str) -> str:
    """Nomor telepon pertama dengan jumlah digit wajar (bukan tahun, NIK, atau nomor rekening)."""
    for match in _PHONE.finditer(text):
        digits = sum(ch.isdigit() for ch in match.group(0))
        if MIN_PHONE_DIGITS <= digits <= MAX_PHONE_DIGITS:
            return " ".join(match.group(0).split())
    return ""


def _short_url(url: str) -> str:
    return _URL_PREFIX.sub("", url).rstrip("/")
