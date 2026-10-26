"""Halaman Cari Lowongan: kata kunci dan 4 filter atas semua lowongan, tanpa LLM dan embedding.

Pencarian ada di ``src.repositories.job_search``; data dibaca lewat ``JobRepository`` dari
``src.containers``. Panel kiri: hasil per ``PAGE_SIZE`` lowongan. Panel kanan: detail lowongan
terpilih, yaitu skill yang diminta (kosakata ``src.cv.skills``) dan deskripsi yang dirapikan
(``src.components.job_description``).

Dengan CV aktif, skill cocok dan kurang dihitung untuk lowongan apa pun (pencocokan kata, sama
dengan Cocokkan CV). Skor hanya untuk lowongan di hasil Cocokkan CV, karena skor butuh kemiripan
profil dari Qdrant.
"""

import logging
import os
from typing import Any

import streamlit as st

from src.components.cv_match import skill_chips, skill_lines
from src.components.job_description import description_to_markdown, escape_markdown
from src.components.navigation import CV_PAGE
from src.containers import get_job_repository
from src.cv.ingest import CV_STATE_KEY
from src.cv.match import JobMatch, JobSkillIndex, load_job_skill_index
from src.cv.skills import build_cv_skills, match_skills
from src.repositories.job_repository import JobRepository
from src.repositories.job_search import (
    JobFilters,
    JobListing,
    filter_options,
    load_listings,
    province_label,
    search_jobs,
)

logger = logging.getLogger(__name__)

PAGE_TITLE = "## Cari lowongan"
PAGE_SIZE = 10
STATE_KEY = "job_search"  # st.session_state: {"filters", "shown", "selected"}
KEYWORD_KEY = "job_search_keyword"
PROVINCE_KEY = "job_search_province"
WORK_TYPE_KEY = "job_search_work_type"
SALARY_KEY = "job_search_salary"
MODE_KEY = "job_search_mode"
SALARY_OPTIONS = (0, 5_000_000, 10_000_000, 15_000_000)
MAX_SKILL_CHIPS = 12
# Label pendek: judul dropdown sudah menyebut filternya, dan di layar 1000 px empat dropdown
# sebaris hanya muat sekitar 8 karakter.
ALL_LABEL = "Semua"
DB_ERROR = "Data lowongan tidak bisa dibaca. Periksa konfigurasi database di file .env."
NO_RESULTS = "Tidak ada lowongan yang cocok. Kurangi kata kunci atau longgarkan filter."
NO_SKILLS = "Deskripsi lowongan ini tidak menyebut skill dari kosakata kami."
SKILL_SOURCE = "Dicocokkan dari kosakata skill di deskripsi, tanpa LLM."
NOT_IN_MATCHES = (
    "Lowongan ini tidak masuk 10 lowongan paling cocok untuk profilmu, jadi skornya tidak "
    "dihitung. Skill di atas dicocokkan dengan kata, tanpa LLM."
)


def render_job_search_page() -> None:
    """Render form pencarian, daftar hasil (kiri), dan detail lowongan terpilih (kanan)."""
    st.markdown(PAGE_TITLE)
    try:
        repository = get_job_repository()
        listings = load_listings(repository)
    except Exception:  # konfigurasi tidak lengkap, file tidak ada, atau query gagal
        logger.exception("Gagal membaca data lowongan untuk Cari Lowongan")
        st.error(DB_ERROR)
        return
    st.caption(f"{len(listings)} lowongan dari data 2025 · pencarian dan filter tanpa LLM")

    state = st.session_state.setdefault(
        STATE_KEY, {"filters": JobFilters(), "shown": PAGE_SIZE, "selected": None}
    )
    _render_form(listings, state)
    results = search_jobs(listings, state["filters"])
    shown = results[: state["shown"]]
    st.caption(results_caption(len(shown), len(results), state["filters"]))
    if not results:
        st.info(NO_RESULTS)
        return
    if state["selected"] not in {job.id for job in shown}:
        state["selected"] = shown[0].id

    list_column, detail_column = st.columns([2, 3], gap="large")
    with list_column:
        for job in shown:
            _render_card(job, selected=job.id == state["selected"], state=state)
        remaining = len(results) - len(shown)
        if remaining > 0 and st.button(
            f"Tampilkan {min(PAGE_SIZE, remaining)} lagi", key="job_search_more"
        ):
            state["shown"] += PAGE_SIZE
            st.rerun()
    selected = next(job for job in shown if job.id == state["selected"])
    with detail_column, st.container(border=True):
        _render_detail(selected, repository)


def _render_form(listings: tuple[JobListing, ...], state: dict[str, Any]) -> None:
    """Form kata kunci dan 4 filter; hasil hanya berubah saat tombol Cari ditekan."""
    options = filter_options(listings)
    salary_counts = {
        value: len(search_jobs(listings, JobFilters(min_salary=value))) for value in SALARY_OPTIONS
    }
    with st.form("job_search_form", border=False):
        keyword_column, button_column = st.columns([6, 1], vertical_alignment="bottom")
        keyword = keyword_column.text_input(
            "Kata kunci",
            key=KEYWORD_KEY,
            placeholder="Posisi, perusahaan, kota, atau skill",
            max_chars=100,
        )
        submitted = button_column.form_submit_button("Cari", type="primary", width="stretch")
        location_column, type_column, salary_column, mode_column = st.columns(4)
        province = location_column.selectbox(
            "Lokasi",
            ["", *(name for name, _ in options.provinces)],
            key=PROVINCE_KEY,
            format_func=counted_label(options.provinces, province_label),
        )
        work_type = type_column.selectbox(
            "Tipe kerja",
            ["", *(name for name, _ in options.work_types)],
            key=WORK_TYPE_KEY,
            format_func=counted_label(options.work_types),
        )
        min_salary = salary_column.selectbox(
            "Gaji",
            SALARY_OPTIONS,
            key=SALARY_KEY,
            format_func=lambda value: salary_option_label(value, salary_counts[value]),
        )
        mode = mode_column.selectbox(
            "Mode kerja",
            ["", *(name for name, _ in options.modes)],
            key=MODE_KEY,
            format_func=counted_label(options.modes),
        )
    if submitted:
        state["filters"] = JobFilters(
            keyword=keyword.strip(),
            province=province or None,
            work_type=work_type or None,
            mode=mode or None,
            min_salary=min_salary,
        )
        state["shown"] = PAGE_SIZE
        state["selected"] = None


def _render_card(job: JobListing, *, selected: bool, state: dict[str, Any]) -> None:
    """Kartu hasil: judul (klik untuk detail), perusahaan, dan baris lokasi · tipe · gaji."""
    with st.container(border=True, gap="xxsmall"):
        if st.button(
            f"**{escape_markdown(job.job_title)}**",
            key=f"job_pick_{job.id}",
            type="tertiary",
            icon=":material/arrow_right:" if selected else None,
        ):
            state["selected"] = job.id
            st.rerun()
        lines = [job.company_name or "", listing_meta(job)]
        st.caption("  \n".join(escape_markdown(line) for line in lines if line))


def _render_detail(job: JobListing, repository: JobRepository) -> None:
    """Judul, badge, kecocokan CV atau skill yang diminta, dan deskripsi."""
    st.markdown(f"### {escape_markdown(job.job_title)}")
    location = " ".join((job.location or "").splitlines()[0].split()) if job.location else ""
    st.caption(escape_markdown(" · ".join(p for p in (job.company_name, location) if p)))
    st.markdown(detail_badges(job))

    cv_state = st.session_state.get(CV_STATE_KEY)
    description = _description(job.id, repository)
    index = _skill_index(_db_path())
    if cv_state is None:
        if st.button("Cek kecocokan dengan CV", key=f"job_fit_{job.id}", type="primary"):
            st.switch_page(CV_PAGE)
        st.divider()
        st.markdown("**Skill yang diminta**")
        st.caption(SKILL_SOURCE)
        skills = required_skills(job.id, index)
        st.markdown(skill_chips(skills, "gray", MAX_SKILL_CHIPS) if skills else NO_SKILLS)
    else:
        st.divider()
        _render_fit(cv_state, job, find_match(cv_state, job.id), description, index)

    st.divider()
    if description:
        st.markdown(description_to_markdown(description))
    else:
        st.caption("Deskripsi tidak tersedia.")


def _render_fit(
    cv_state: dict[str, Any],
    job: JobListing,
    match: JobMatch | None,
    description: str | None,
    index: JobSkillIndex | None,
) -> None:
    """Kecocokan dengan CV aktif: skor dari Cocokkan CV, atau skill saja jika tidak ada."""
    st.markdown("**Kecocokan dengan CV-mu**")
    if match is not None:
        st.markdown(f"#### {match.score:.0%}")
        for line in skill_lines(match):
            st.markdown(line)
        return
    if index is None:
        st.caption(NO_SKILLS)
        return
    skills = match_skills(
        build_cv_skills(cv_state["profile"].skills),
        index.vocab.get(job.id, frozenset()),
        description,
        frequency=index.frequency,
    )
    if not skills.required:
        st.caption(NO_SKILLS)
    else:
        st.markdown(f"Skill cocok: {skill_chips(list(skills.matched), 'green') or 'belum ada'}")
        st.markdown(f"Skill kurang: {skill_chips(list(skills.missing), 'orange') or 'tidak ada'}")
    st.caption(NOT_IN_MATCHES)


def _description(job_id: int, repository: JobRepository) -> str | None:
    """Deskripsi dari database; ``None`` jika gagal (detail lain tetap tampil)."""
    try:
        jobs = repository.get_jobs_by_ids([job_id])
    except Exception:  # deskripsi hanya bagian dari panel detail
        logger.exception("Gagal mengambil deskripsi lowongan untuk Cari Lowongan")
        return None
    return jobs[0].get("job_description") if jobs else None


def _skill_index(db_path: str) -> JobSkillIndex | None:
    """Indeks skill per lowongan (di-cache per proses); ``None`` jika database gagal."""
    try:
        return load_job_skill_index(db_path)
    except Exception:  # skill hanya bagian dari panel detail
        logger.exception("Gagal membangun indeks skill untuk Cari Lowongan")
        return None


def _db_path() -> str:
    """Path SQLite untuk indeks skill ``src.cv.match``, yang masih membaca database lewat path."""
    return os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")


# --- Helper tanpa Streamlit (bisa dites langsung) -------------------------------


def find_match(cv_state: dict[str, Any], job_id: int) -> JobMatch | None:
    """Lowongan ``job_id`` di hasil Cocokkan CV untuk profil saat ini; ``None`` jika tidak ada."""
    return next((match for match in cv_state["matches"].matches if match.job_id == job_id), None)


def short_salary(salary_min: int | None, salary_max: int | None) -> str:
    """Gaji ringkas untuk kartu: "Rp8–10 jt", "Rp7,5 jt", atau "Gaji tidak dicantumkan"."""
    values = [value for value in (salary_min, salary_max) if value]
    if not values:
        return "Gaji tidak dicantumkan"

    def million(value: int) -> str:
        return f"{value / 1_000_000:.1f}".removesuffix(".0").replace(".", ",")

    low, high = min(values), max(values)
    if low == high:
        return f"Rp{million(low)} jt"
    return f"Rp{million(low)}–{million(high)} jt"


def listing_meta(job: JobListing) -> str:
    """Baris kartu "lokasi · tipe kerja · mode · gaji"; bagian yang kosong dilewati."""
    parts = (job.place, job.work_type, job.mode, short_salary(job.salary_min, job.salary_max))
    return " · ".join(part for part in parts if part)


def detail_badges(job: JobListing) -> str:
    """Badge Markdown Streamlit: tipe kerja, mode, gaji, dan ID lowongan."""
    salary = short_salary(job.salary_min, job.salary_max)
    salary_color = "green" if job.salary_min or job.salary_max else "orange"
    badges = [
        f":gray-badge[{_badge(job.work_type)}]" if job.work_type else "",
        f":blue-badge[{job.mode}]" if job.mode else "",
        f":{salary_color}-badge[{salary}]",
        f":gray-badge[ID {job.id}]",
    ]
    return " ".join(badge for badge in badges if badge)


def required_skills(job_id: int, index: JobSkillIndex | None) -> list[str]:
    """Skill kosakata yang disebut lowongan, paling umum di seluruh lowongan lebih dulu."""
    if index is None:
        return []
    names = index.vocab.get(job_id, frozenset())
    return sorted(names, key=lambda name: (-index.frequency.get(name, 0), name.lower()))


def results_caption(shown: int, total: int, filters: JobFilters) -> str:
    """Keterangan jumlah hasil, urutan, dan efek filter gaji."""
    parts = [f"Menampilkan {shown} dari {total} lowongan"]
    if filters.keyword:
        parts.append("urut relevansi kata kunci")
    if filters.min_salary:
        parts.append("lowongan tanpa info gaji tidak ditampilkan")
    return " · ".join(parts)


def counted_label(counts: list[tuple[str, int]], label: Any = None) -> Any:
    """``format_func`` selectbox: ``ALL_LABEL`` untuk "", selain itu "Nama (jumlah)".

    Pilihan "semua" memakai "" (bukan ``None``) agar selectbox tidak menampilkan tombol hapus.
    """
    lookup = dict(counts)

    def format_option(value: str) -> str:
        if not value:
            return ALL_LABEL
        return f"{label(value) if label else value} ({lookup.get(value, 0)})"

    return format_option


def salary_option_label(value: int, count: int) -> str:
    """Label pilihan gaji, misalnya "Minimal Rp5 juta (152)"."""
    if not value:
        return ALL_LABEL
    return f"Minimal Rp{value // 1_000_000} juta ({count})"


def _badge(text: str) -> str:
    # Kurung siku menutup direktif badge; "$" memicu LaTeX di markdown Streamlit.
    return text.replace("[", "(").replace("]", ")").replace("$", r"\$")
