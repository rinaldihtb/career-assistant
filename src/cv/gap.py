"""Skill gap dari lowongan teratas hasil Cocokkan CV, tanpa LLM dan tanpa embedding.

Input cukup ``MatchResult.matches``: setiap ``JobMatch`` sudah membawa ``matched_skills``,
``missing_skills``, dan ``score``, jadi tidak ada query ulang. Hitungannya sekitar
100 operasi, sehingga tidak di-cache.

Urutan skill:
1. Jumlah lowongan yang menyebut skill ("7 dari 10"), supaya angka yang tampil selalu sesuai
   urutan. Pada top-10 kelima CV set eval Cocokkan CV, urutan berbobot skor lowongan tidak
   pernah menaruh skill dengan hitungan lebih kecil di atas hitungan lebih besar; bedanya
   hanya pada skill yang seri.
2. Seri dipecah dengan total skor lowongan yang menyebutnya (skill dari lowongan yang lebih
   cocok lebih dulu). Seri sering terjadi: 4 skill "1 dari 10" di cv01, 6 di cv05.
3. Lalu urutan kemunculan pertama: lowongan urut skor, dan ``missing_skills`` per lowongan
   sudah urut frekuensi skill di 473 lowongan.

Tidak ada ambang minimal: skill gap di set eval jarang (tertinggi 2-4 dari 10), dan ambang
">= 2 lowongan" mengosongkan skill gap cv04 dan menyisakan satu skill untuk cv03.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.cv.match import JobMatch

GAP_LIMIT = 10


@dataclass(frozen=True)
class SkillCount:
    """Satu skill dan berapa lowongan teratas yang menyebutnya.

    Attributes:
        name: Nama skill (nama kosakata, atau skill CV di luar kosakata untuk ``have``).
        jobs: Jumlah lowongan yang menyebut skill ini.
        score_sum: Total skor lowongan tersebut, pemecah seri.
        job_ids: ID lowongan yang menyebutnya, urut skor; rujukan angka "x dari N".
    """

    name: str
    jobs: int
    score_sum: float
    job_ids: tuple[int, ...] = ()


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
    missing = _count((match.missing_skills, match.score, match.job_id) for match in matches)
    have = _count((match.matched_skills, match.score, match.job_id) for match in matches)
    return SkillGap(
        missing=tuple(missing[:limit]),
        have=tuple(have[:limit]),
        jobs=len(matches),
        jobs_without_skills=sum(1 for match in matches if match.skill_portion is None),
        missing_total=len(missing),
        have_total=len(have),
    )


def _count(rows: Iterable[tuple[list[str], float, int]]) -> list[SkillCount]:
    ids: dict[str, list[int]] = {}
    scores: dict[str, float] = {}
    for skills, score, job_id in rows:
        for name in dict.fromkeys(skills):  # satu lowongan dihitung sekali per skill
            ids.setdefault(name, []).append(job_id)
            scores[name] = scores.get(name, 0.0) + score
    first_seen = {name: index for index, name in enumerate(ids)}
    ordered = sorted(ids, key=lambda name: (-len(ids[name]), -scores[name], first_seen[name]))
    return [
        SkillCount(name=name, jobs=len(ids[name]), score_sum=scores[name], job_ids=tuple(ids[name]))
        for name in ordered
    ]
