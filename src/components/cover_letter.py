"""Halaman surat lamaran untuk satu lowongan hasil Cocokkan CV.

Halaman dibuka dari tombol "Buat surat lamaran" di kartu lowongan; ID lowongannya disimpan di
``st.session_state["cv"]["letter_job"]``. Saat dibuka pertama kali untuk versi profil dan
lowongan itu, surat dibuat dengan 1 panggilan LLM (``src.cv.letter``) dan disimpan di
``st.session_state["cv"]["letters"][job_id]``. "Buat ulang" memanggil LLM lagi dengan bahasa dan
panjang yang dipilih. Teks surat (tanggal, penerima, perihal, isi, penutup) bisa diedit; editan
disimpan di entri yang sama. "Unduh .pdf" dan "Unduh .docx" memakai teks yang sudah diedit dan
baris kontak kop surat, dan baru dibuat saat tombol diklik.

Surat dan profil hanya disimpan di ``st.session_state``. Pesan error LLM tidak di-log dan
tidak ditampilkan, karena bisa memuat potongan data CV.
"""

import logging
from datetime import date
from functools import partial
from typing import Any

import streamlit as st

from src.components.cv_match import (
    LETTER_JOB_KEY,
    load_dependencies,
    match_meta,
    score_formula,
    skill_lines,
)
from src.components.job_description import description_to_markdown, escape_markdown
from src.components.navigation import CV_PAGE
from src.containers import traced_llm
from src.cv.ingest import CV_STATE_KEY
from src.cv.letter import (
    LANGUAGE_NAMES,
    LENGTH_LABELS,
    LETTER_LENGTHS,
    NAME_PLACEHOLDER,
    CoverLetterResult,
    LetterSettings,
    generate_cover_letter,
    letter_file_name,
    letter_header,
    letter_to_text,
    signer_name,
    word_count,
)
from src.cv.letter_file import letter_to_docx, letter_to_pdf
from src.cv.match import JobMatch
from src.cv.profile import CVProfile
from src.repositories.job_repository import get_jobs_by_ids, sqlite_path

logger = logging.getLogger(__name__)

LETTERS_KEY = "letters"  # di dalam st.session_state["cv"]: {job_id: entri surat}
LANGUAGE_KEY = "cv_letter_language"
WORDS_KEY = "cv_letter_words"
CONTACT_KEY = "letter_contact"  # di dalam st.session_state["cv"]: baris kontak kop surat
CONTACT_SEPARATOR = "  ·  "
DEFAULT_SETTINGS = LetterSettings()
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF_MIME = "application/pdf"
TEXT_HEIGHT = 520
NO_CV = (
    "Belum ada CV aktif. Upload CV di halaman Cocokkan CV, lalu pilih lowongan yang ingin "
    "dibuatkan surat lamaran."
)
NO_JOB = "Pilih lowongan di halaman Cocokkan CV, lalu klik Buat surat lamaran di kartunya."
JOB_GONE = (
    "Lowongan ini tidak ada lagi di hasil pencocokan profil terbaru. Pilih lowongan lain di "
    "halaman Cocokkan CV."
)
DEPENDENCY_ERROR = (
    "Surat lamaran belum bisa dibuat karena konfigurasi LLM belum lengkap. Periksa file .env."
)
LETTER_FAILED = "Surat lamaran gagal dibuat. Klik Buat ulang untuk mencoba lagi."
LETTER_NOTE = (
    "1 panggilan LLM dari profil CV dan lowongan ini; teks CV tidak dikirim. Skill dan "
    "pengalaman di surat hanya dari profil. Nama, kontak, tanggal, penerima, dan perihal "
    "disusun tanpa LLM."
)
CONTACT_HELP = "Baris kecil di bawah nama pada kop surat. Diambil dari CV; bisa diubah."
NAME_MISSING = f"Nama tidak terbaca dari CV. Ganti {NAME_PLACEHOLDER} di akhir surat."


def render_cover_letter_page() -> None:
    """Render halaman: detail lowongan dalam tab, dengan tab Surat lamaran terbuka."""
    state = st.session_state.get(CV_STATE_KEY)
    if state is None:
        _go_to_cv_match(NO_CV)
        return
    job_id = state.get(LETTER_JOB_KEY)
    if job_id is None:
        _go_to_cv_match(NO_JOB)
        return
    match = find_match(state, job_id)
    if match is None:
        _go_to_cv_match(JOB_GONE)
        return

    if st.button(
        "Kembali ke hasil", icon=":material/arrow_back:", type="tertiary", key="letter_back"
    ):
        st.switch_page(CV_PAGE)
    st.markdown(f"## {escape_markdown(match.job_title)}")
    st.caption(escape_markdown(match_meta(match)))
    description = _description(job_id)
    labels = ["Ringkasan", f"Kecocokan CV · {match.score:.0%}", "Surat lamaran"]
    summary, fit, letter = st.tabs(labels, default=labels[2])
    with summary:
        if description:
            st.markdown(description_to_markdown(description))
        else:
            st.caption("Deskripsi tidak tersedia.")
    with fit:
        _render_fit(match)
    with letter:
        _render_letter(state, match, description)


def _go_to_cv_match(message: str) -> None:
    st.info(message)
    if st.button("Buka Cocokkan CV", key="letter_to_cv"):
        st.switch_page(CV_PAGE)


def _render_fit(match: JobMatch) -> None:
    """Skor kecocokan dan skill, sama dengan kartu di Cocokkan CV."""
    st.markdown(f"### {match.score:.0%}")
    st.progress(min(1.0, max(0.0, match.score)))
    for line in skill_lines(match):
        st.markdown(line)
    st.caption(score_formula())


def _render_letter(state: dict[str, Any], match: JobMatch, description: str | None) -> None:
    """Surat yang bisa diedit (kiri) dan pengaturan serta unduh (kanan).

    Tanpa surat untuk versi profil ini, surat dibuat dulu (1 LLM) dengan pengaturan aktif.
    """
    entry = cached_letter(state, match.job_id)
    if entry is None:
        _generate(state, match, description, current_settings(st.session_state))
        return

    profile = state["profile"]
    result: CoverLetterResult = entry["result"]
    left, right = st.columns([3, 2], gap="large")
    text = ""
    with left:
        if result.letter is None:
            st.warning(LETTER_FAILED)
        else:
            key = f"cv_letter_text_{state['version']}_{match.job_id}_{entry['round']}"
            text = st.text_area(
                "Surat lamaran",
                value=entry.get("text") or letter_text(entry, match, profile),
                height=TEXT_HEIGHT,
                key=key,
                label_visibility="collapsed",
                on_change=_save_edit,
                args=(entry, key),
            )
            st.caption(f"{word_count(text)} kata · bisa diedit sebelum diunduh.")
            if signer_name(profile.full_name) == NAME_PLACEHOLDER:
                st.warning(NAME_MISSING)
    with right:
        with st.container(border=True):
            st.markdown("#### Pengaturan")
            st.selectbox(
                "Bahasa",
                options=list(LANGUAGE_NAMES),
                index=list(LANGUAGE_NAMES).index(DEFAULT_SETTINGS.language),
                format_func=LANGUAGE_NAMES.get,
                key=LANGUAGE_KEY,
            )
            st.selectbox(
                "Panjang",
                options=LETTER_LENGTHS,
                index=LETTER_LENGTHS.index(DEFAULT_SETTINGS.words),
                format_func=LENGTH_LABELS.get,
                key=WORDS_KEY,
            )
            regenerate = st.button("Buat ulang", key="cv_letter_regenerate")
            contact_key = f"cv_letter_contact_{state['version']}"
            contact = st.text_input(
                "Kontak di kop surat",
                value=state.get(CONTACT_KEY, contact_line(profile)),
                help=CONTACT_HELP,
                key=contact_key,
                on_change=_save_contact,
                args=(state, contact_key),
            )
            file_args = {"author": _author(profile.full_name), "contact": contact}
            with st.container(horizontal=True):
                st.download_button(
                    "Unduh .pdf",
                    data=partial(letter_to_pdf, text, **file_args) if text else b"",
                    file_name=letter_file_name(match, "pdf"),
                    mime=PDF_MIME,
                    on_click="ignore",
                    type="primary",
                    disabled=not text,
                    key="cv_letter_download_pdf",
                )
                st.download_button(
                    "Unduh .docx",
                    data=partial(letter_to_docx, text, **file_args) if text else b"",
                    file_name=letter_file_name(match),
                    mime=DOCX_MIME,
                    on_click="ignore",
                    disabled=not text,
                    key="cv_letter_download",
                )
        st.caption(LETTER_NOTE)
        st.caption(letter_caption(entry))
    if regenerate:
        _generate(state, match, description, current_settings(st.session_state))


def _generate(
    state: dict[str, Any], match: JobMatch, description: str | None, settings: LetterSettings
) -> None:
    """Panggil LLM sekali, simpan hasil (juga jika gagal) untuk versi profil ini, lalu rerun."""
    try:
        deps = load_dependencies()
    except Exception:  # error konfigurasi bisa KeyError, koneksi Qdrant, dll.
        logger.exception("Gagal memuat dependency surat lamaran")
        st.error(DEPENDENCY_ERROR)
        return
    llm = traced_llm(deps.llm, "cv_letter")
    with st.spinner("Menulis surat lamaran..."):
        result = generate_cover_letter(
            state["profile"], match, description, llm=llm, settings=settings
        )
    letters = state.setdefault(LETTERS_KEY, {})
    previous = letters.get(match.job_id)
    letters[match.job_id] = {
        "version": state["version"],
        "settings": settings,
        "result": result,
        "round": previous["round"] + 1 if previous else 1,
    }
    st.rerun()


def _save_edit(entry: dict[str, Any], key: str) -> None:
    """Simpan editan ke entri surat, supaya tetap ada setelah pindah halaman."""
    entry["text"] = st.session_state[key]


def _save_contact(state: dict[str, Any], key: str) -> None:
    """Simpan baris kontak kop surat untuk semua surat di sesi ini."""
    state[CONTACT_KEY] = st.session_state[key]


def _description(job_id: int) -> str | None:
    """Deskripsi lowongan dari database; ``None`` jika gagal (surat tetap bisa dibuat)."""
    try:
        jobs = get_jobs_by_ids(sqlite_path(), [job_id])
    except Exception:  # deskripsi hanya pelengkap prompt dan tab Ringkasan
        logger.exception("Gagal mengambil deskripsi lowongan untuk surat lamaran")
        return None
    return jobs[0].get("job_description") if jobs else None


# --- Helper tanpa Streamlit (bisa dites langsung) -------------------------------


def find_match(state: dict[str, Any], job_id: int) -> JobMatch | None:
    """Lowongan ``job_id`` di hasil pencocokan versi profil saat ini; ``None`` jika tidak ada."""
    return next((m for m in state["matches"].matches if m.job_id == job_id), None)


def cached_letter(state: dict[str, Any], job_id: int) -> dict[str, Any] | None:
    """Entri surat untuk lowongan ini; ``None`` jika belum dibuat atau profil sudah berubah."""
    entry = state.get(LETTERS_KEY, {}).get(job_id)
    if entry is None or entry["version"] != state["version"]:
        return None
    return entry


def current_settings(session: Any) -> LetterSettings:
    """Pengaturan dari widget; default jika widget belum pernah tampil."""
    return LetterSettings(
        language=session.get(LANGUAGE_KEY, DEFAULT_SETTINGS.language),
        words=session.get(WORDS_KEY, DEFAULT_SETTINGS.words),
    )


def letter_text(
    entry: dict[str, Any], match: JobMatch, profile: CVProfile, today: date | None = None
) -> str:
    """Teks awal surat: header (tanggal, penerima, perihal), salam, isi dari LLM, penutup, nama."""
    language = entry["settings"].language
    header = letter_header(
        match, language=language, city=profile.location, today=today or date.today()
    )
    return letter_to_text(
        entry["result"].letter,
        company=match.company_name,
        name=signer_name(profile.full_name),
        language=language,
        header=header,
    )


def contact_line(profile: CVProfile) -> str:
    """Baris kontak awal untuk kop: lokasi profil lalu kontak hasil regex dari CV."""
    parts = [" ".join((profile.location or "").split()), *profile.contacts]
    return CONTACT_SEPARATOR.join(part for part in parts if part)


def letter_caption(entry: dict[str, Any]) -> str:
    """Pengaturan surat yang sedang tampil."""
    settings: LetterSettings = entry["settings"]
    return f"{LANGUAGE_NAMES[settings.language]} · {LENGTH_LABELS[settings.words]}."


def _author(full_name: str | None) -> str:
    name = signer_name(full_name)
    return "" if name == NAME_PLACEHOLDER else name
