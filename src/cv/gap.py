"""Skill gap dari lowongan teratas hasil Cocokkan CV. Python murni: 0 LLM, 0 embedding.

Input cukup ``MatchResult.matches`` yang sudah ada di ``st.session_state["cv"]``: setiap
``JobMatch`` sudah membawa ``matched_skills``, ``missing_skills``, dan ``score``, jadi tidak
perlu query ulang. Hitungannya sekitar 100 operasi, sehingga tidak di-cache.

Rumus urutan (diukur pada top-10 cv01..cv05 set eval Cocokkan CV, run
cv-c1d-rescored-20261022-0808):
1. Jumlah lowongan yang menyebut skill ("7 dari 10"). Dibandingkan urutan berbobot skor
   lowongan, bobot skor TIDAK PERNAH menaruh skill dengan hitungan lebih kecil di atas skill
   dengan hitungan lebih besar di kelima CV; bedanya hanya pada skill yang seri. Jumlah
   dipakai sebagai kunci utama supaya angka "x dari 10" yang tampil selalu sesuai urutan.
2. Seri dipecah dengan total skor lowongan yang menyebutnya (skill dari lowongan yang lebih
   cocok lebih dulu). Seri sering terjadi: 4 skill "1 dari 10" di cv01, 6 di cv05.
3. Lalu urutan kemunculan pertama: lowongan urut skor, dan ``missing_skills`` per lowongan
   sudah urut frekuensi skill di 473 lowongan.

Tanpa ambang minimal: skill gap set eval jarang (tertinggi 2-4 dari 10). Ambang ">= 2
lowongan" akan mengosongkan skill gap cv04 dan menyisakan satu skill untuk cv03.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from cv.match import JobMatch

GAP_LIMIT = 10


@dataclass(frozen=True)
class SkillCount:
    """Satu skill dan berapa lowongan teratas yang menyebutnya.

    Attributes:
        name: Nama skill (nama kosakata, atau skill CV di luar kosakata untuk ``have``).
        jobs: Jumlah lowongan yang menyebut skill ini.
        score_sum: Total skor lowongan tersebut, pemecah seri.
    """

    name: str
    jobs: int
    score_sum: float


@dataclass(frozen=True)
class SkillGap:
    """Ringkasan skill dari lowongan teratas.

    Attributes:
        missing: Skill yang disebut lowongan tetapi belum ada di profil, maksimal ``limit``.
        have: Skill profil yang disebut lowongan, maksimal ``limit``.
        jobs: Jumlah lowongan yang dihitung (penyebut "x dari N").
        jobs_without_skills: Lowongan yang tidak menyebut skill kosakata sama sekali.
        missing_total: Jumlah skill kurang yang berbeda sebelum dipotong ``limit``.
        have_total: Jumlah skill cocok yang berbeda sebelum dipotong ``limit``.
    """

    missing: tuple[SkillCount, ...]
    have: tuple[SkillCount, ...]
    jobs: int
    jobs_without_skills: int
    missing_total: int
    have_total: int


def compute_skill_gap(matches: Sequence[JobMatch], *, limit: int = GAP_LIMIT) -> SkillGap:
    """Agregasi skill kurang dan skill cocok atas lowongan hasil pencocokan.

    Args:
        matches: ``MatchResult.matches`` (urut skor menurun, biasanya 10 lowongan).
        limit: Jumlah skill maksimal per daftar.

    Returns:
        ``SkillGap``; daftar kosong jika ``matches`` kosong.
    """
    missing = _count((match.missing_skills, match.score) for match in matches)
    have = _count((match.matched_skills, match.score) for match in matches)
    return SkillGap(
        missing=tuple(missing[:limit]),
        have=tuple(have[:limit]),
        jobs=len(matches),
        jobs_without_skills=sum(1 for match in matches if match.skill_portion is None),
        missing_total=len(missing),
        have_total=len(have),
    )


def _count(rows: Iterable[tuple[list[str], float]]) -> list[SkillCount]:
    jobs: dict[str, int] = {}
    scores: dict[str, float] = {}
    for skills, score in rows:
        for name in dict.fromkeys(skills):  # satu lowongan dihitung sekali per skill
            jobs[name] = jobs.get(name, 0) + 1
            scores[name] = scores.get(name, 0.0) + score
    first_seen = {name: index for index, name in enumerate(jobs)}
    ordered = sorted(jobs, key=lambda name: (-jobs[name], -scores[name], first_seen[name]))
    return [SkillCount(name=name, jobs=jobs[name], score_sum=scores[name]) for name in ordered]
