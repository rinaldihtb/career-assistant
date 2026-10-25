"""Profil CV terstruktur dari teks CV: 1 panggilan LLM lalu perhitungan deterministik.

Pembagian kerja:
- LLM (structured output) hanya MENYALIN fakta: nama, posisi tujuan, lokasi, pendidikan
  tertinggi, skill, dan daftar pengalaman beserta periodenya.
- Python MENGHITUNG lama pengalaman dari periode itu: periode yang tumpang-tindih
  digabung dan "present" berarti bulan ini. LLM tidak tahu tanggal hari ini dan mudah
  salah menjumlahkan periode yang tumpang-tindih; hitungan Python bisa dites.

Kontrak sama dengan agent: dependency (LLM) disuntikkan, fungsi tidak pernah melempar
exception, kegagalan dilaporkan lewat ``CVProfileResult.error``. Profil hanya dikembalikan,
tidak disimpan ke disk atau log.
"""

import re
from collections.abc import Sequence
from datetime import date
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.prompts import CV_PROFILE_SYSTEM, CV_PROFILE_USER
from src.models.agent import LLMUsage

EducationLevel = Literal["SMA/SMK", "D3", "D4", "S1", "S2", "S3"]
MAX_SKILLS = 30
PRESENT_WORDS = {"present", "sekarang", "saat ini", "now", "current"}
_PERIOD = re.compile(r"^(\d{4})(?:-(\d{1,2}))?$")


class Experience(BaseModel):
    """Satu pengalaman kerja, magang, atau freelance seperti tertulis di CV."""

    title: str = Field(description="Jabatan persis seperti di CV.")
    company: str | None = Field(description="Nama perusahaan; null jika tidak tertulis.")
    start: str | None = Field(description="Mulai: YYYY-MM, atau YYYY jika bulan tidak tertulis.")
    end: str | None = Field(
        description="Selesai: YYYY-MM, YYYY, atau 'present' jika masih berjalan; null jika tidak tertulis."
    )


class CVExtraction(BaseModel):
    """Skema yang dikirim ke LLM sebagai JSON schema.

    Semua field wajib (tanpa default) agar skema valid untuk structured output mode strict,
    sama seperti ``Plan`` di supervisor.
    """

    full_name: str | None = Field(description="Nama lengkap pemilik CV.")
    target_role: str | None = Field(description="Posisi yang dituju.")
    location: str | None = Field(description="Kota domisili.")
    education_level: EducationLevel | None = Field(description="Jenjang pendidikan tertinggi.")
    education_major: str | None = Field(description="Jurusan pada jenjang tertinggi.")
    skills: list[str] = Field(description="Keahlian yang tertulis eksplisit, maksimal 30.")
    experiences: list[Experience] = Field(description="Pekerjaan, magang, dan freelance.")


class CVProfile(BaseModel):
    """Profil yang dipakai aplikasi; setiap field boleh dikoreksi pengguna di UI.

    Attributes:
        experience_months: Total bulan pengalaman hasil hitungan Python. ``0`` jika CV
            tidak mencantumkan pengalaman; ``None`` jika ada pengalaman tetapi tidak ada
            periode yang bisa dihitung.
        experiences: Data mentah dari LLM, disimpan agar hitungan bisa diperiksa.
    """

    full_name: str | None = None
    target_role: str | None = None
    location: str | None = None
    education_level: EducationLevel | None = None
    education_major: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience_months: int | None = None
    experiences: list[Experience] = Field(default_factory=list)

    @property
    def experience_label(self) -> str:
        """Rentang pengalaman seperti di iklan lowongan, misalnya "1-3 tahun"."""
        return experience_label(self.experience_months)


class CVProfileResult(BaseModel):
    """Hasil ``extract_cv_profile``: ``profile`` terisi jika berhasil, selain itu ``error``."""

    profile: CVProfile | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    error: str | None = None


def extract_cv_profile(
    cv_text: str, *, llm: BaseChatModel, today: date | None = None
) -> CVProfileResult:
    """Ekstrak profil dari teks CV dengan tepat satu panggilan LLM.

    Args:
        cv_text: Teks hasil ``cv.pdf_text.extract_cv_text``.
        llm: Chat model; dipanggil lewat ``with_structured_output(CVExtraction)``.
        today: Tanggal acuan untuk periode "present". Default hari ini; diisi di test
            agar hasil pasti.

    Returns:
        ``CVProfileResult``. Teks kosong: 0 panggilan LLM dan ``error`` terisi.
        Tidak pernah melempar exception.
    """
    usage = LLMUsage()
    if not cv_text.strip():
        return CVProfileResult(usage=usage, error="Teks CV kosong")

    messages = [
        SystemMessage(CV_PROFILE_SYSTEM),
        HumanMessage(CV_PROFILE_USER.format(cv_text=cv_text)),
    ]
    extractor = llm.with_structured_output(CVExtraction, include_raw=True)
    try:
        output = extractor.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - tipe error provider LLM sangat beragam
        return CVProfileResult(usage=usage, error=f"LLM error: {exc}")

    usage.add(output["raw"], step="cv_profile")
    parsed: CVExtraction | None = output["parsed"]
    if parsed is None:
        return CVProfileResult(
            usage=usage, error=f"Output LLM tidak valid: {output['parsing_error']}"
        )
    return CVProfileResult(profile=build_profile(parsed, today or date.today()), usage=usage)


def build_profile(extraction: CVExtraction, today: date) -> CVProfile:
    """Rapikan hasil LLM dan hitung lama pengalaman. Tanpa LLM."""
    return CVProfile(
        full_name=_clean(extraction.full_name),
        target_role=_clean(extraction.target_role),
        location=_clean(extraction.location),
        education_level=extraction.education_level,
        education_major=_clean(extraction.education_major),
        skills=unique_skills(extraction.skills),
        experience_months=experience_months(extraction.experiences, today),
        experiences=extraction.experiences,
    )


def unique_skills(skills: Sequence[str]) -> list[str]:
    """Buang skill kosong dan duplikat (tanpa membedakan huruf besar), maksimal ``MAX_SKILLS``.

    Urutan kemunculan pertama dipertahankan karena biasanya mencerminkan prioritas di CV.
    """
    seen: set[str] = set()
    result = []
    for skill in skills:
        name = " ".join(skill.split())
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            result.append(name)
    return result[:MAX_SKILLS]


def experience_months(experiences: Sequence[Experience], today: date) -> int | None:
    """Total bulan pengalaman; periode yang tumpang-tindih dihitung sekali.

    Aturan periode (inklusif per bulan): ``YYYY`` sebagai awal berarti Januari, sebagai
    akhir berarti Desember; akhir ``present`` atau setelah bulan ini dipotong ke bulan
    ini. Periode tanpa awal atau akhir, atau yang akhirnya sebelum awalnya, dilewati
    karena panjangnya tidak bisa dipastikan.

    Returns:
        0 jika tidak ada pengalaman; ``None`` jika ada pengalaman tetapi tidak satu pun
        periodenya bisa dihitung.
    """
    if not experiences:
        return 0
    now = today.year * 12 + today.month - 1
    spans = []
    for exp in experiences:
        start = _month_index(exp.start, is_end=False)
        end = now if _is_present(exp.end) else _month_index(exp.end, is_end=True)
        if start is None or end is None:
            continue
        end = min(end, now)
        if end >= start:
            spans.append((start, end))
    if not spans:
        return None

    spans.sort()
    total = 0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start + 1
            current_start, current_end = start, end
    return total + current_end - current_start + 1


def experience_label(months: int | None) -> str:
    """Kelompokkan bulan pengalaman ke rentang yang lazim di iklan lowongan."""
    if months is None:
        return "Tidak diketahui"
    if months < 12:
        return "0-1 tahun"
    if months < 36:
        return "1-3 tahun"
    if months < 60:
        return "3-5 tahun"
    return "5+ tahun"


def _month_index(value: str | None, *, is_end: bool) -> int | None:
    match = _PERIOD.match((value or "").strip())
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else (12 if is_end else 1)
    if not 1 <= month <= 12:
        return None
    return year * 12 + month - 1


def _is_present(value: str | None) -> bool:
    return (value or "").strip().casefold() in PRESENT_WORDS


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split()) or None
