"""Halaman Cocokkan CV (Figma 02a kosong, 02b hasil).

Alur:
1. Kosong (02a): upload PDF dan centang persetujuan, baru tombol "Baca CV & cocokkan" aktif.
   Klik: ``ingest_cv`` (1 LLM) lalu ``match_jobs`` (1 embedding, 0 LLM); hasilnya ditulis
   sekali ke ``st.session_state["cv"]`` = {profile, version, matches, ...}.
2. Hasil (02b): profil terbaca bisa dikoreksi. "Cocokkan ulang" menaikkan ``version``;
   embedding ulang hanya jika teks profil yang di-embed berubah (koreksi lokasi tidak).
3. "Ganti" mengosongkan state dan uploader.

CV dan profil hanya di ``st.session_state`` (memori sesi), tidak ditulis ke disk atau log.
Preferensi singkat di Figma 02a (posisi, lokasi, tipe kerja) belum diterapkan: ``match_jobs``
belum memakai filter itu dan efeknya belum diukur di set eval Cocokkan CV.
"""

import logging
import os
import typing
from dataclasses import dataclass
from typing import Any

import streamlit as st

from agents.rag_agent import format_salary
from components.job_description import description_to_markdown, escape_markdown
from cv.ingest import CV_STATE_KEY, IngestResult, ingest_cv
from cv.match import JobMatch, MatchConfig, MatchResult, match_jobs, profile_query_text
from cv.profile import MAX_SKILLS, CVProfile, EducationLevel, unique_skills
from repositories.job_repository import get_jobs_by_ids

logger = logging.getLogger(__name__)

PAGE_TITLE = "## Cocokkan CV dengan lowongan"
UPLOADER_ROUND_KEY = "cv_uploader_round"  # dinaikkan saat "Ganti" agar uploader kosong lagi
CONSENT_KEY = "cv_consent"
CONSENT_TEXT = (
    "Saya setuju CV diproses LLM (OpenAI) selama sesi ini saja; tidak disimpan ke disk atau log."
)
MAX_UPLOAD_MB = 5  # sama dengan pdf_text.MAX_FILE_BYTES; file lebih besar ditolak sebelum upload
MAX_CHIPS = 3
EDUCATION_LEVELS: tuple[str, ...] = typing.get_args(EducationLevel)
DEPENDENCY_ERROR = (
    "Cocokkan CV belum bisa dijalankan karena konfigurasi LLM, embedding, atau Qdrant belum "
    "lengkap. Periksa file .env."
)
MATCH_FAILED = (
    "Profil sudah terbaca, tetapi pencarian lowongan gagal. Klik Cocokkan ulang untuk mencoba lagi."
)
NO_MATCHES = "Belum ada lowongan yang mirip dengan profil ini. Periksa target posisi dan skill."
EMPTY_RESULTS = (
    "Belum ada hasil. Upload CV di kiri untuk melihat 10 lowongan paling cocok, skornya, dan "
    "skill yang masih kurang."
)
REVIEW_HINT = (
    "Periksa dulu: LLM bisa salah membaca CV. Ubah isian, tambah atau hapus skill, lalu "
    "Cocokkan ulang."
)


@dataclass(frozen=True)
class CVDependencies:
    """Dependency untuk membaca dan mencocokkan CV."""

    llm: Any
    embeddings: Any
    vector_store: Any
    db_path: str


@st.cache_resource(show_spinner=False)
def load_dependencies() -> CVDependencies:
    """LLM, embedding, dan Qdrant dari ``config``, dibuat sekali per proses.

    Import di dalam fungsi: jika ``.env`` belum lengkap, hanya aksi di halaman ini yang
    gagal dengan pesan jelas. ``st.cache_resource`` tidak menyimpan exception, jadi setelah
    ``.env`` diperbaiki pemanggilan berikutnya mencoba lagi.
    """
    from config.database import VECTOR_DB
    from config.llm import EMBEDDING, LLM_MODEL

    return CVDependencies(
        llm=LLM_MODEL,
        embeddings=EMBEDDING,
        vector_store=VECTOR_DB,
        db_path=_db_path(),
    )


def render_cv_match_page() -> None:
    """Render halaman: panel kiri (upload atau profil terbaca), panel kanan (hasil)."""
    st.markdown(PAGE_TITLE)
    st.caption(
        "Upload CV PDF-mu. Profil diekstrak sekali oleh LLM, bisa kamu koreksi, lalu "
        "dicocokkan dengan semua lowongan di data kami."
    )
    left, right = st.columns([2, 3], gap="large")
    with left:
        if st.session_state.get(CV_STATE_KEY) is None:
            _render_upload()
        else:
            _render_profile_panel(st.session_state[CV_STATE_KEY])
    # Dibaca ulang: "Cocokkan ulang" di panel kiri bisa sudah memperbarui state.
    state = st.session_state.get(CV_STATE_KEY)
    with right:
        st.markdown("#### 10 lowongan paling cocok")
        st.caption(score_formula())
        if state is None:
            st.info(EMPTY_RESULTS)
            if st.button("Cari lowongan tanpa CV", key="cv_browse"):
                st.switch_page("pages/jobsearch.page.py")
        else:
            _render_results(state)


# --- Panel kiri ----------------------------------------------------------------


def _render_upload() -> None:
    """02a: uploader, persetujuan, dan tombol yang aktif hanya jika keduanya terpenuhi."""
    round_ = st.session_state.get(UPLOADER_ROUND_KEY, 0)
    uploaded = st.file_uploader(
        "CV (PDF)",
        type=["pdf"],
        max_upload_size=MAX_UPLOAD_MB,
        key=f"cv_uploader_{round_}",
        help="PDF berteks, maksimal 5 MB. PDF hasil scan atau foto ditolak.",
    )
    # wrap=True: tanpa ini label persetujuan terpotong "...", padahal harus terbaca utuh.
    consent = st.checkbox(CONSENT_TEXT, key=CONSENT_KEY, wrap=True)
    clicked = st.button(
        "Baca CV & cocokkan",
        type="primary",
        disabled=uploaded is None or not consent,
        key="cv_read",
    )
    if clicked and uploaded is not None and consent:
        _ingest_and_match(uploaded)


def _ingest_and_match(uploaded: Any) -> None:
    """Baca CV (1 LLM), cocokkan (1 embedding), simpan state sekali, lalu rerun ke 02b."""
    deps = _dependencies()
    if deps is None:
        return
    with st.spinner("Membaca CV..."):
        ingest = ingest_cv(uploaded.getvalue(), llm=deps.llm)
    if ingest.profile is None:
        st.error(ingest.error)
        return
    with st.spinner("Mencocokkan dengan lowongan..."):
        matches = _match(ingest.profile, deps)
    st.session_state[CV_STATE_KEY] = new_cv_state(
        file_name=uploaded.name, file_size=uploaded.size, ingest=ingest, matches=matches
    )
    st.rerun()


def _render_profile_panel(state: dict[str, Any]) -> None:
    """02b kiri: kartu file, form koreksi profil, dan "Cocokkan ulang"."""
    profile: CVProfile = state["profile"]
    version = state["version"]
    with st.container(border=True):
        name_column, button_column = st.columns([4, 1], vertical_alignment="center")
        name_column.markdown(f"**{escape_markdown(state['file_name'])}**")
        name_column.caption(
            f"{size_label(state['file_size'])} · {state['pages']} halaman · terbaca"
        )
        button_column.button("Ganti", on_click=reset_cv, key="cv_replace")

    # Key memakai version: setelah koreksi diterapkan, widget dibuat ulang dari profil baru.
    with st.form("cv_profile_form"):
        st.markdown("**Profil terbaca**")
        target_role = st.text_input(
            "Target posisi", value=profile.target_role or "", key=f"cv_target_{version}"
        )
        st.text_input(
            "Pengalaman",
            value=profile.experience_label,
            disabled=True,
            help="Dihitung dari periode pengalaman di CV.",
            key=f"cv_experience_{version}",
        )
        level_column, major_column = st.columns([1, 2])
        level = level_column.selectbox(
            "Pendidikan",
            EDUCATION_LEVELS,
            index=EDUCATION_LEVELS.index(profile.education_level)
            if profile.education_level
            else None,
            placeholder="Pilih",
            key=f"cv_level_{version}",
        )
        major = major_column.text_input(
            "Jurusan", value=profile.education_major or "", key=f"cv_major_{version}"
        )
        location = st.text_input(
            "Lokasi", value=profile.location or "", key=f"cv_location_{version}"
        )
        skills = st.multiselect(
            "Skill",
            options=profile.skills,
            default=profile.skills,
            accept_new_options=True,
            max_selections=MAX_SKILLS,
            placeholder="Ketik untuk menambah skill",
            key=f"cv_skills_{version}",
        )
        st.warning(REVIEW_HINT)
        submitted = st.form_submit_button("Cocokkan ulang", type="primary")
    if submitted:
        edited = edit_profile(
            profile,
            target_role=target_role,
            education_level=level,
            education_major=major,
            location=location,
            skills=skills,
        )
        _apply_edits(state, edited)
    st.caption(usage_caption(st.session_state[CV_STATE_KEY]))


def _apply_edits(state: dict[str, Any], edited: CVProfile) -> None:
    """Terapkan koreksi: version naik jika profil berubah; embedding hanya jika perlu.

    Pencocokan juga diulang tanpa perubahan jika pencocokan sebelumnya gagal (tombol
    "Cocokkan ulang" sekaligus tombol coba lagi).
    """
    previous_failed = state["matches"].error is not None
    if edited == state["profile"] and not previous_failed:
        st.toast("Tidak ada perubahan pada profil.")
        return
    updated = {**state, "profile": edited}
    if edited != state["profile"]:
        updated["version"] = state["version"] + 1
    if previous_failed or needs_rematch(state["profile"], edited):
        deps = _dependencies()
        if deps is None:
            return
        with st.spinner("Mencocokkan ulang..."):
            matches = _match(edited, deps)
        updated["matches"] = matches
        updated["embedding_calls"] = state["embedding_calls"] + matches.embedding_calls
    st.session_state[CV_STATE_KEY] = updated


def reset_cv() -> None:
    """Callback "Ganti": hapus CV aktif, kosongkan uploader, dan minta persetujuan lagi."""
    st.session_state.pop(CV_STATE_KEY, None)
    st.session_state[UPLOADER_ROUND_KEY] = st.session_state.get(UPLOADER_ROUND_KEY, 0) + 1
    st.session_state[CONSENT_KEY] = False


# --- Panel kanan ---------------------------------------------------------------


def _render_results(state: dict[str, Any]) -> None:
    """02b kanan: kartu lowongan urut skor."""
    result: MatchResult = state["matches"]
    if result.error:
        st.warning(MATCH_FAILED)
        return
    if not result.matches:
        st.info(NO_MATCHES)
        return
    descriptions = _descriptions([match.job_id for match in result.matches])
    for match in result.matches:
        _render_match_card(match, descriptions.get(match.job_id))


def _render_match_card(match: JobMatch, description: str | None) -> None:
    """Kartu: judul, meta, skor, chip skill, detail, dan tombol surat (nonaktif)."""
    with st.container(border=True):
        info_column, score_column = st.columns([4, 1])
        info_column.markdown(f"**{escape_markdown(match.job_title)}**")
        info_column.caption(escape_markdown(match_meta(match)))
        score_column.markdown(f"### {match.score:.0%}")
        st.progress(min(1.0, max(0.0, match.score)))
        for line in skill_lines(match):
            st.markdown(line)
        with st.expander("Lihat detail lowongan"):
            if description:
                st.markdown(description_to_markdown(description))
            else:
                st.caption("Deskripsi tidak tersedia.")
        st.button(
            "Buat surat lamaran",
            key=f"cv_letter_{match.job_id}",
            disabled=True,
            help="Segera hadir: surat lamaran dari profil CV dan lowongan ini.",
        )


# --- Dependency dan data -------------------------------------------------------


def _dependencies() -> CVDependencies | None:
    """``load_dependencies`` dengan pesan error yang ramah; ``None`` jika gagal."""
    try:
        return load_dependencies()
    except Exception:  # error konfigurasi bisa KeyError, koneksi Qdrant, dll.
        logger.exception("Gagal memuat dependency Cocokkan CV")
        st.error(DEPENDENCY_ERROR)
        return None


def _match(profile: CVProfile, deps: CVDependencies) -> MatchResult:
    result = match_jobs(
        profile,
        embeddings=deps.embeddings,
        vector_store=deps.vector_store,
        db_path=deps.db_path,
    )
    if result.error:
        # Isi error dari provider embedding/Qdrant, bukan data CV.
        logger.warning("Pencocokan CV gagal: %s", result.error)
    return result


def _descriptions(job_ids: list[int]) -> dict[int, str]:
    """Deskripsi untuk "Lihat detail"; kosong jika database gagal (kartu tetap tampil)."""
    try:
        jobs = get_jobs_by_ids(_db_path(), job_ids)
    except Exception:  # detail hanya pelengkap kartu
        logger.exception("Gagal mengambil deskripsi lowongan untuk kartu Cocokkan CV")
        return {}
    return {job["id"]: job["job_description"] for job in jobs if job.get("job_description")}


def _db_path() -> str:
    return os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")


# --- Helper tanpa Streamlit (bisa dites langsung) -------------------------------


def new_cv_state(
    *, file_name: str, file_size: int, ingest: IngestResult, matches: MatchResult
) -> dict[str, Any]:
    """State CV aktif untuk ``st.session_state["cv"]``; ``version`` mulai dari 1."""
    return {
        "profile": ingest.profile,
        "version": 1,
        "matches": matches,
        "file_name": file_name,
        "file_size": file_size,
        "pages": ingest.pages,
        "usage": ingest.usage,
        "embedding_calls": matches.embedding_calls,
    }


def edit_profile(
    profile: CVProfile,
    *,
    target_role: str | None,
    education_level: str | None,
    education_major: str | None,
    location: str | None,
    skills: list[str],
) -> CVProfile:
    """Profil baru dari isian form: spasi dirapikan, isian kosong menjadi ``None``, skill
    unik (tanpa membedakan huruf besar) dan maksimal ``MAX_SKILLS``. Field lain tetap."""
    return profile.model_copy(
        update={
            "target_role": _clean(target_role),
            "education_level": education_level or None,
            "education_major": _clean(education_major),
            "location": _clean(location),
            "skills": unique_skills(skills),
        }
    )


def needs_rematch(old: CVProfile, new: CVProfile) -> bool:
    """Embedding ulang hanya jika teks yang di-embed berubah.

    Skill, posisi, judul pengalaman, dan pendidikan ada di teks itu; lokasi dan nama tidak
    (lihat ``profile_query_text``). Mengoreksi lokasi saja tidak memakan biaya embedding.
    """
    return profile_query_text(old) != profile_query_text(new)


def score_formula(config: MatchConfig | None = None) -> str:
    """Penjelasan skor yang jujur terhadap kode: bobot dari ``MatchConfig``, skill lewat
    pencocokan kata (bukan SQL atau LLM), skill wajib dan opsional tidak dibedakan."""
    weight = (config or MatchConfig()).semantic_weight
    return (
        f"Skor = {weight:.0%} kemiripan profilmu dengan deskripsi lowongan (Qdrant) + "
        f"{1 - weight:.0%} porsi skill yang disebut lowongan dan ada di profilmu "
        "(pencocokan kata, tanpa LLM; skill wajib dan opsional dihitung sama)."
    )


def match_meta(match: JobMatch) -> str:
    """Baris "perusahaan · lokasi · tipe kerja · gaji"; bagian yang kosong dilewati."""
    location = " ".join((match.location or "").split())
    salary = (
        format_salary(match.salary_min, match.salary_max)
        if match.salary_min or match.salary_max
        else None
    )
    parts = (match.company_name, location, match.work_type, salary)
    return " · ".join(part for part in parts if part)


def skill_chips(skills: list[str], color: str, limit: int = MAX_CHIPS) -> str:
    """Badge markdown Streamlit (``:green-badge[...]``), maksimal ``limit`` lalu "+N"."""
    chips = [f":{color}-badge[{_badge_text(skill)}]" for skill in skills[:limit]]
    if len(skills) > limit:
        chips.append(f":gray-badge[+{len(skills) - limit}]")
    return " ".join(chips)


def skill_lines(match: JobMatch) -> list[str]:
    """Baris "Skill cocok" dan "Skill kurang" untuk kartu.

    Lowongan yang tidak menyebut skill kosakata (``skill_portion is None``) mendapat satu
    keterangan: skornya hanya dari kemiripan deskripsi.
    """
    if match.skill_portion is None:
        return ["Lowongan ini tidak menyebut skill spesifik; skor hanya dari kemiripan deskripsi."]
    matched = skill_chips(match.matched_skills, "green") or "belum ada"
    missing = skill_chips(match.missing_skills, "orange") or "tidak ada"
    return [f"Skill cocok: {matched}", f"Skill kurang: {missing}"]


def size_label(size: int) -> str:
    """Ukuran file untuk kartu: "182 KB" atau "1,2 MB"."""
    if size < 1024 * 1024:
        return f"{max(1, round(size / 1024))} KB"
    return f"{size / (1024 * 1024):.1f} MB".replace(".", ",")


def usage_caption(state: dict[str, Any]) -> str:
    """Biaya yang sudah dipakai CV ini: LLM saat membaca, embedding saat mencocokkan."""
    usage = state["usage"]
    tokens = f"{usage.total_tokens:,}".replace(",", ".")
    return (
        f"Biaya: {usage.llm_calls} panggilan LLM ({tokens} token) untuk membaca CV · "
        f"{state['embedding_calls']} embedding untuk mencocokkan · versi profil "
        f"{state['version']}."
    )


def _clean(text: str | None) -> str | None:
    cleaned = " ".join((text or "").split())
    return cleaned or None


def _badge_text(skill: str) -> str:
    # Kurung siku menutup direktif badge; "$" memicu LaTeX di markdown Streamlit.
    return skill.replace("[", "(").replace("]", ")").replace("$", r"\$")
