"""Agent CV: jawaban dari hasil Cocokkan CV yang sudah ada di sesi. 0 LLM, 0 embedding, 0 query.

Dua jenis sub-tugas dari planner (``SubTask.agent``):

- ``cv_match``: ``CV_MATCH_LIMIT`` lowongan teratas dari ``MatchResult.matches`` (sudah urut
  skor), dengan skor dan skill yang cocok. Barisnya berformat ``- [ID:x] Judul - Perusahaan
  (Lokasi): catatan``, sama dengan RAG/SQL, sehingga UI membuat kartu dan riwayat planner
  menyimpan ID-nya.
- ``cv_gap``: hasil ``compute_skill_gap`` atas lowongan yang sama, jadi angkanya sama persis
  dengan halaman Skill Gap.
- ``cv_role``: skill gap dan lowongan teratas untuk posisi lain yang disebut pengguna. Hasil
  pencocokannya (``match_role``, 1 embedding) dihitung supervisor dan diteruskan lewat
  ``matches``; jawabannya tetap template.

Jawaban disusun dari template, bukan LLM: data sudah dihitung saat CV dicocokkan, dan teks
pertanyaan tidak mengubah hasil. Profil hanya dipakai untuk label posisi dan pengalaman.
"""

from typing import Literal

from src.agents.prompts import (
    CV_FAILED_MESSAGE,
    CV_GAP_HAVE,
    CV_GAP_INTRO,
    CV_GAP_LINE,
    CV_GAP_NONE,
    CV_GAP_PLAN,
    CV_MATCH_INTRO,
    CV_MATCH_LINE,
    CV_MATCH_SKILLS,
    CV_NO_MATCHES_MESSAGE,
    CV_ROLE_EMPTY_MESSAGE,
    CV_ROLE_FAILED_MESSAGE,
    CV_ROLE_GAP_INTRO,
    CV_ROLE_GAP_NONE,
    CV_ROLE_HAVE,
    CV_ROLE_JOBS,
    CV_ROLE_LINE,
    CV_ROLE_NO_SKILLS,
    CV_ROLE_SKILLS,
    CV_SOURCE,
    CV_SOURCE_WITH_LABEL,
)
from src.cv.gap import compute_skill_gap
from src.cv.match import JobMatch, MatchConfig, MatchResult
from src.cv.profile import CVProfile
from src.models.agent import AgentResult

CVKind = Literal["cv_match", "cv_gap", "cv_role"]

# Halaman Cocokkan CV menampilkan 10 lowongan; di chat 5 cukup sebagai ringkasan, dan
# kartu lowongan di chat lebih tinggi daripada baris teks.
CV_MATCH_LIMIT = 5
CV_GAP_LIMIT = 5
CV_HAVE_LIMIT = 3
MAX_MATCHED_SKILLS = 3


def run_cv_agent(
    *,
    kind: CVKind,
    profile: CVProfile | None,
    matches: MatchResult | None,
    role: str = "",
) -> AgentResult:
    """Jawab satu sub-tugas CV dari hasil pencocokan yang sudah ada.

    Untuk ``cv_match`` dan ``cv_gap`` teks pertanyaan dari planner tidak dibutuhkan: hasilnya
    sama untuk setiap kalimat yang meminta jenis jawaban yang sama.

    Args:
        kind: ``cv_match`` (lowongan teratas), ``cv_gap`` (skill kurang), atau ``cv_role``
            (skill kurang dan lowongan teratas untuk posisi lain).
        profile: Profil CV aktif, untuk label posisi dan pengalaman.
        matches: Hasil ``match_jobs`` yang tersimpan di sesi; untuk ``cv_role``, hasil
            ``match_role`` posisi itu.
        role: Nama posisi lain (khusus ``cv_role``).

    Returns:
        ``AgentResult`` dengan ``source="cv"`` dan ``usage`` kosong. ``job_ids`` berisi
        lowongan yang ditampilkan (``cv_match`` dan ``cv_role``). Tanpa hasil pencocokan,
        jawaban berupa ajakan membuka Cocokkan CV. Tidak pernah melempar exception.
    """
    try:
        if kind == "cv_role":
            return role_answer(_one_line(role), matches)
        if matches is None or matches.error or not matches.matches:
            return AgentResult(answer=CV_NO_MATCHES_MESSAGE, source="cv")
        if kind == "cv_gap":
            return AgentResult(answer=gap_answer(matches.matches), source="cv")
        shown = matches.matches[:CV_MATCH_LIMIT]
        return AgentResult(
            answer=match_answer(profile, shown, total=len(matches.matches)),
            source="cv",
            job_ids=[match.job_id for match in shown],
        )
    except Exception as exc:  # noqa: BLE001 - kontrak agent: tidak pernah melempar exception
        return AgentResult(answer=CV_FAILED_MESSAGE, source="cv", error=f"CV agent error: {exc}")


def match_answer(profile: CVProfile | None, shown: list[JobMatch], *, total: int) -> str:
    """Teks jawaban ``cv_match``: pembuka (jumlah, rumus skor), lalu satu baris per lowongan."""
    weight = MatchConfig().semantic_weight
    label = profile_label(profile)
    lines = [
        CV_MATCH_INTRO.format(
            source=CV_SOURCE_WITH_LABEL.format(label=label) if label else CV_SOURCE,
            count=len(shown),
            total=total,
            semantic=f"{weight:.0%}",
            skill=f"{1 - weight:.0%}",
        )
    ]
    for match in shown:
        skills = match.matched_skills[:MAX_MATCHED_SKILLS]
        lines.append(
            CV_MATCH_LINE.format(
                id=match.job_id,
                title=_one_line(match.job_title),
                company=_one_line(match.company_name),
                location=_one_line(match.location),
                score=f"{match.score:.0%}",
                skills=CV_MATCH_SKILLS.format(skills=", ".join(skills)) if skills else "",
            )
        )
    return "\n".join(lines)


def gap_answer(matches: list[JobMatch]) -> str:
    """Teks jawaban ``cv_gap``: skill kurang "x dari N", skill yang dimiliki, rujukan rencana."""
    gap = compute_skill_gap(matches, limit=CV_GAP_LIMIT)
    if not gap.missing:
        parts = [CV_GAP_NONE.format(jobs=gap.jobs)]
    else:
        lines = [CV_GAP_INTRO.format(jobs=gap.jobs)]
        lines += [
            CV_GAP_LINE.format(name=skill.name, count=skill.jobs, jobs=gap.jobs)
            for skill in gap.missing
        ]
        parts = ["\n".join(lines)]
    if gap.have:
        have = ", ".join(
            f"{skill.name} ({skill.jobs} dari {gap.jobs})" for skill in gap.have[:CV_HAVE_LIMIT]
        )
        parts.append(CV_GAP_HAVE.format(skills=have))
    if gap.missing:
        parts.append(CV_GAP_PLAN)
    return "\n\n".join(parts)


def role_answer(role: str, matches: MatchResult | None) -> AgentResult:
    """Jawaban ``cv_role``: skill gap posisi itu, skill CV yang terpakai, lalu lowongan teratas.

    ``error`` diisi jika pencarian gagal; pesan error asli tidak masuk ke ``answer``.
    """
    if matches is None or matches.error:
        error = matches.error if matches is not None else "Hasil pencocokan posisi kosong"
        return AgentResult(
            answer=CV_ROLE_FAILED_MESSAGE.format(role=role), source="cv", error=error
        )
    if not matches.matches:
        return AgentResult(answer=CV_ROLE_EMPTY_MESSAGE.format(role=role), source="cv")
    gap = compute_skill_gap(matches.matches, limit=CV_GAP_LIMIT)
    if gap.missing:
        lines = [CV_ROLE_GAP_INTRO.format(role=role, jobs=gap.jobs)]
        lines += [
            CV_GAP_LINE.format(name=skill.name, count=skill.jobs, jobs=gap.jobs)
            for skill in gap.missing
        ]
        parts = ["\n".join(lines)]
    else:
        parts = [CV_ROLE_GAP_NONE.format(role=role, jobs=gap.jobs)]
    if gap.have:
        have = ", ".join(
            f"{skill.name} ({skill.jobs} dari {gap.jobs})" for skill in gap.have[:CV_HAVE_LIMIT]
        )
        parts.append(CV_ROLE_HAVE.format(skills=have))
    shown = matches.matches[:CV_MATCH_LIMIT]
    jobs = [CV_ROLE_JOBS.format(count=len(shown), role=role)]
    for match in shown:
        skills = match.matched_skills[:MAX_MATCHED_SKILLS]
        jobs.append(
            CV_ROLE_LINE.format(
                id=match.job_id,
                title=_one_line(match.job_title),
                company=_one_line(match.company_name),
                location=_one_line(match.location),
                skills=CV_ROLE_SKILLS.format(skills=", ".join(skills))
                if skills
                else CV_ROLE_NO_SKILLS,
            )
        )
    parts.append("\n".join(jobs))
    return AgentResult(
        answer="\n\n".join(parts), source="cv", job_ids=[match.job_id for match in shown]
    )


def profile_label(profile: CVProfile | None) -> str:
    """Label singkat profil, misalnya "Data Analyst · 1-3 tahun"; kosong jika tidak ada data."""
    if profile is None:
        return ""
    parts = [_one_line(profile.target_role)] if profile.target_role else []
    if profile.experience_months is not None:
        parts.append(profile.experience_label)
    return " · ".join(part for part in parts if part != "-")


def _one_line(text: str | None) -> str:
    """Rapikan spasi dan baris baru (lokasi di data bisa memuat "\\n(Hibrid)"); kosong -> "-"."""
    return " ".join((text or "").split()) or "-"
