"""Rencana belajar 4 minggu dari skill gap, dengan tepat 1 panggilan LLM (structured output).

Prompt hanya berisi profil ringkas (posisi tujuan, label pengalaman, pendidikan, skill) dan
skill gap dari ``cv.gap.compute_skill_gap``; teks CV, nama, lokasi, dan riwayat kerja tidak
dikirim. Biaya terukur (``make sample-plan``): 560-656 token masuk + 201-251 keluar.

LLM disuntikkan, fungsi tidak pernah melempar exception, dan kegagalan ada di
``LearningPlanResult.error``. Halaman Skill Gap meng-cache hasil per ``version`` profil.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.prompts import CV_PLAN_SYSTEM, CV_PLAN_USER
from cv.gap import SkillGap
from cv.profile import CVProfile
from models.agent import LLMUsage

PLAN_WEEKS = 4


class PlanWeek(BaseModel):
    """Satu minggu rencana belajar."""

    title: str = Field(description="Fokus minggu ini, maksimal 8 kata.")
    skills: list[str] = Field(
        description="Nama skill dari daftar skill gap yang dilatih minggu ini, persis sama."
    )
    tasks: list[str] = Field(description="1-3 kegiatan konkret yang selesai dalam seminggu.")


class LearningPlan(BaseModel):
    """Skema structured output. Semua field wajib (tanpa default), sama seperti
    ``CVExtraction``, agar valid untuk structured output mode strict."""

    weeks: list[PlanWeek] = Field(description="Tepat 4 minggu, urut dari minggu 1.")


class LearningPlanResult(BaseModel):
    """Hasil ``generate_learning_plan``: ``plan`` terisi jika berhasil, selain itu ``error``."""

    plan: LearningPlan | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    error: str | None = None


def plan_prompt_input(profile: CVProfile, gap: SkillGap) -> tuple[str, str]:
    """Teks ``{profile}`` dan ``{gap}`` untuk ``CV_PLAN_USER``. Tanpa nama dan lokasi.

    Returns:
        ``(profil, skill_gap)``; baris profil yang kosong dilewati.
    """
    education = " ".join(p for p in (profile.education_level, profile.education_major) if p)
    lines = [
        f"Posisi tujuan: {profile.target_role}" if profile.target_role else "",
        f"Pengalaman: {profile.experience_label}",
        f"Pendidikan: {education}" if education else "",
        f"Skill di profil: {', '.join(profile.skills)}" if profile.skills else "",
    ]
    gap_lines = [f"- {skill.name}: {skill.jobs} dari {gap.jobs} lowongan" for skill in gap.missing]
    return "\n".join(line for line in lines if line), "\n".join(gap_lines)


def generate_learning_plan(
    profile: CVProfile, gap: SkillGap, *, llm: BaseChatModel
) -> LearningPlanResult:
    """Susun rencana belajar dengan tepat satu panggilan LLM.

    Args:
        profile: Profil CV aktif (mungkin sudah dikoreksi pengguna).
        gap: Hasil ``compute_skill_gap`` dari lowongan teratas profil ini.
        llm: Chat model; dipanggil lewat ``with_structured_output(LearningPlan)``.

    Returns:
        ``LearningPlanResult``. Tanpa skill gap: 0 panggilan LLM dan ``error`` terisi.
        Lebih dari ``PLAN_WEEKS`` minggu dipotong; nol minggu dianggap gagal.
        Tidak pernah melempar exception.
    """
    usage = LLMUsage()
    if not gap.missing:
        return LearningPlanResult(usage=usage, error="Tidak ada skill gap")

    profile_text, gap_text = plan_prompt_input(profile, gap)
    messages = [
        SystemMessage(CV_PLAN_SYSTEM),
        HumanMessage(CV_PLAN_USER.format(profile=profile_text, gap=gap_text)),
    ]
    planner = llm.with_structured_output(LearningPlan, include_raw=True)
    try:
        output = planner.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - tipe error provider LLM sangat beragam
        return LearningPlanResult(usage=usage, error=f"LLM error: {exc}")

    usage.add(output["raw"], step="cv_plan")
    parsed: LearningPlan | None = output["parsed"]
    if parsed is None:
        return LearningPlanResult(
            usage=usage, error=f"Output LLM tidak valid: {output['parsing_error']}"
        )
    if not parsed.weeks:
        return LearningPlanResult(usage=usage, error="Rencana kosong")
    return LearningPlanResult(plan=LearningPlan(weeks=parsed.weeks[:PLAN_WEEKS]), usage=usage)


def plan_to_markdown(plan: LearningPlan, gap: SkillGap, target_role: str | None) -> str:
    """Isi file "Unduh rencana (.md)": checklist per minggu dan skill gap dasarnya. Tanpa LLM."""
    title = f"# Rencana belajar {len(plan.weeks)} minggu"
    lines = [title + (f": {target_role}" if target_role else ""), ""]
    if gap.missing:
        counted = ", ".join(f"{skill.name} ({skill.jobs}/{gap.jobs})" for skill in gap.missing)
        lines += [f"Skill gap dari {gap.jobs} lowongan paling cocok: {counted}", ""]
    for number, week in enumerate(plan.weeks, start=1):
        lines.append(f"## Minggu {number}: {week.title}")
        if week.skills:
            lines.append(f"Skill: {', '.join(week.skills)}")
        lines += [f"- [ ] {task}" for task in week.tasks]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
