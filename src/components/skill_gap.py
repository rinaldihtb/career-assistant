"""Halaman Skill Gap: skill yang paling sering diminta lowongan teratas dan rencana belajar.

Halaman membaca CV aktif dari ``st.session_state["cv"]``; tanpa CV, halaman mengarahkan ke
Cocokkan CV. Skill gap dihitung dari ``matches`` yang sudah ada (``src.cv.gap``: 0 LLM,
0 embedding, sekitar 100 operasi, tidak di-cache). Rencana belajar dibuat dengan 1 panggilan
LLM saat tombol diklik, lalu di-cache di ``st.session_state["cv"]["plan"]`` per ``version``
profil; "Ganti" CV menghapusnya bersama state CV.

Tombol "Tanya Asisten" membuka Asisten Karier dengan pertanyaan skill gap yang langsung dijawab
(1 panggilan LLM planner); Asisten membaca CV yang sama dari ``st.session_state``, sehingga
pertanyaan lanjutan (misalnya lowongan yang meminta skill itu di kota tertentu) punya konteks.

Bar memakai ``st.progress`` agar urutan tampil sama persis dengan urutan hitungan. Halaman
tidak menampilkan estimasi kenaikan skor: menambah 2 skill gap teratas hanya menaikkan skor
rata-rata top-10 sebesar 1,0-2,5 poin di set eval Cocokkan CV (porsi skill berbobot 40% dan
dibagi jumlah skill disebut + 2).
"""

import logging
from typing import Any

import streamlit as st

from src.components.assistant_chat import PENDING_KEY
from src.components.cv_match import load_dependencies
from src.components.job_description import escape_markdown
from src.components.navigation import ASSISTANT_PAGE, CV_PAGE
from src.containers import traced_llm
from src.cv.gap import SkillCount, SkillGap, compute_skill_gap
from src.cv.ingest import CV_STATE_KEY
from src.cv.match import MatchResult
from src.cv.plan import LearningPlanResult, PlanWeek, generate_learning_plan, plan_to_markdown
from src.helpers.format import format_usage

logger = logging.getLogger(__name__)

PAGE_TITLE = "## Skill gap & rencana belajar"
PLAN_STATE_KEY = "plan"  # di dalam st.session_state["cv"]: {"version": int, "result": ...}
NO_CV = (
    "Belum ada CV aktif. Upload CV di halaman Cocokkan CV dulu; skill gap dihitung dari "
    "10 lowongan yang paling cocok dengan profilmu."
)
NO_MATCHES = "Belum ada hasil pencocokan untuk CV ini. Buka Cocokkan CV lalu klik Cocokkan ulang."
NO_GAP = "Semua skill yang disebut lowongan teratas sudah ada di profilmu."
NO_HAVE = "Belum ada skill profilmu yang disebut lowongan teratas."
DEPENDENCY_ERROR = (
    "Rencana belajar belum bisa dibuat karena konfigurasi LLM belum lengkap. Periksa file .env."
)
PLAN_FAILED = "Rencana belajar gagal dibuat. Coba lagi beberapa saat lagi."
ASSISTANT_QUESTION = "Skill apa yang perlu kupelajari dulu?"
PLAN_HINT = (
    "Rencana 4 minggu disusun dari skill gap di kiri dengan 1 panggilan LLM. Yang dikirim "
    "hanya posisi tujuan, pengalaman, pendidikan, dan daftar skill; teks CV tidak dikirim."
)


def render_skill_gap_page() -> None:
    """Render halaman: skill gap (kiri) dan rencana belajar (kanan)."""
    st.markdown(PAGE_TITLE)
    state = st.session_state.get(CV_STATE_KEY)
    if state is None:
        _go_to_cv_match(NO_CV)
        return
    result: MatchResult = state["matches"]
    if result.error or not result.matches:
        _go_to_cv_match(NO_MATCHES)
        return

    gap = compute_skill_gap(result.matches)
    st.caption(
        f"Dibandingkan dengan {gap.jobs} lowongan paling cocok untuk CV-mu "
        f"({escape_markdown(state['file_name'])}) · versi profil {state['version']}."
    )
    left, right = st.columns([3, 2], gap="large")
    with left:
        _render_counts("Paling sering diminta, belum ada di CV", gap.missing, gap.jobs, NO_GAP)
        st.caption(gap_caption(gap))
        _render_counts("Sudah kamu punya", gap.have, gap.jobs, NO_HAVE)
    with right:
        _render_plan(state, gap)
        if st.button(
            "Tanya Asisten", icon=":material/chat:", width="stretch", key="gap_to_assistant"
        ):
            st.session_state[PENDING_KEY] = ASSISTANT_QUESTION
            st.switch_page(ASSISTANT_PAGE)


def _go_to_cv_match(message: str) -> None:
    st.info(message)
    if st.button("Buka Cocokkan CV", key="gap_to_cv"):
        st.switch_page(CV_PAGE)


def _render_counts(title: str, skills: tuple[SkillCount, ...], jobs: int, empty: str) -> None:
    """Satu kartu bar "x dari N"; ``st.progress`` agar urutan persis sama dengan rumus."""
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if not skills:
            st.caption(empty)
        for skill in skills:
            st.progress(skill.jobs / jobs, text=count_label(skill, jobs))


def _render_plan(state: dict[str, Any], gap: SkillGap) -> None:
    """Kartu rencana: tombol (1 LLM) atau rencana yang sudah di-cache untuk versi ini."""
    with st.container(border=True):
        st.markdown("#### Rencana 4 minggu")
        cached = cached_plan(state)
        if cached is None or cached.plan is None:
            st.caption(PLAN_HINT)
            if cached is not None:
                st.warning(PLAN_FAILED)
            clicked = st.button(
                "Buat rencana belajar" if cached is None else "Coba lagi",
                type="primary",
                disabled=not gap.missing,
                key="gap_plan",
            )
            if clicked:
                _generate(state, gap)
            return

        version = state["version"]
        for number, week in enumerate(cached.plan.weeks, start=1):
            st.checkbox(week_label(number, week), key=f"gap_week_{version}_{number}", wrap=True)
            if week.tasks:
                st.caption("\n".join(f"- {escape_markdown(task)}" for task in week.tasks))
        st.download_button(
            "Unduh rencana (.md)",
            data=plan_to_markdown(cached.plan, gap, state["profile"].target_role),
            file_name="rencana-belajar.md",
            mime="text/markdown",
            on_click="ignore",
            type="primary",
            key="gap_download",
        )
        st.caption(plan_usage_caption(cached))


def _generate(state: dict[str, Any], gap: SkillGap) -> None:
    """Panggil LLM sekali, simpan hasil (juga jika gagal) untuk versi profil ini, lalu rerun.

    Pesan error LLM tidak di-log dan tidak ditampilkan: bisa memuat potongan data profil.
    """
    try:
        deps = load_dependencies()
    except Exception:  # error konfigurasi bisa KeyError, koneksi Qdrant, dll.
        logger.exception("Gagal memuat dependency rencana belajar")
        st.error(DEPENDENCY_ERROR)
        return
    llm = traced_llm(deps.llm, "cv_plan")
    with st.spinner("Menyusun rencana belajar..."):
        result = generate_learning_plan(state["profile"], gap, llm=llm)
    state[PLAN_STATE_KEY] = {"version": state["version"], "result": result}
    st.rerun()


# --- Helper tanpa Streamlit (bisa dites langsung) -------------------------------


def cached_plan(state: dict[str, Any]) -> LearningPlanResult | None:
    """Rencana untuk versi profil saat ini; ``None`` jika belum dibuat atau profil berubah."""
    entry = state.get(PLAN_STATE_KEY)
    if entry is None or entry["version"] != state["version"]:
        return None
    return entry["result"]


def count_label(skill: SkillCount, jobs: int) -> str:
    """Teks bar, misalnya "Power BI · 7 dari 10 · ID 14, 43" (ID lowongan sebagai rujukan)."""
    label = f"{escape_markdown(skill.name)} · {skill.jobs} dari {jobs}"
    if skill.job_ids:
        label += " · ID " + ", ".join(str(job_id) for job_id in skill.job_ids)
    return label


def gap_caption(gap: SkillGap) -> str:
    """Cara hitung dan apa yang tidak ditampilkan, supaya angka di atasnya bisa dipercaya."""
    parts = [
        f"Dihitung dengan Python dari {gap.jobs} lowongan teratas, tanpa LLM.",
        "Urutan: jumlah lowongan, lalu skor kecocokan lowongan.",
    ]
    hidden = gap.missing_total - len(gap.missing)
    if hidden > 0:
        parts.append(f"{hidden} skill lain diminta lebih sedikit lowongan.")
    if gap.jobs_without_skills:
        parts.append(f"{gap.jobs_without_skills} lowongan tidak menyebut skill spesifik.")
    return " ".join(parts)


def week_label(number: int, week: PlanWeek) -> str:
    """Label checkbox minggu, misalnya "Minggu 1 — Power BI dasar"."""
    return f"Minggu {number} — {escape_markdown(week.title)}"


def plan_usage_caption(result: LearningPlanResult) -> str:
    """Biaya rencana yang sudah dibuat untuk versi profil ini."""
    return f"Biaya rencana ini: {format_usage(result.usage)}."
