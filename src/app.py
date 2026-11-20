"""Streamlit application entry point."""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Env dimuat sebelum halaman jalan: alamat dan kunci API dibaca sebelum config di-import.
load_dotenv()
# The entry point lives in ``src/`` while pages use absolute ``src.*`` imports.
# Make the repository root importable in every runtime, including Streamlit Cloud.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Diimpor setelah PROJECT_ROOT masuk sys.path karena modulnya memakai import absolut src.*.
from src.components.brand import FAVICON, LOGO_MARK, LOGO_WORDMARK

st.set_page_config(
    page_title="bisakerja",
    page_icon=str(FAVICON),
    layout="wide",
    initial_sidebar_state="locked",
)
# Wordmark di sidebar terbuka, logomark saat sidebar tertutup (layar sempit).
st.logo(str(LOGO_WORDMARK), size="large", icon_image=str(LOGO_MARK))

home_page = st.Page(
    "pages/home_page.py",
    title="Beranda",
    icon="🏠",
    default=True,
)
job_search_page = st.Page(
    "pages/jobsearch_page.py",
    title="Cari Lowongan",
    icon="🔎",
)

cv_match_page = st.Page(
    "pages/cvmatch_page.py",
    title="Cocokkan CV",
    icon="📄",
)

skill_gap_page = st.Page(
    "pages/skillgap_page.py",
    title="Skill Gap",
    icon="📊",
)

letter_page = st.Page(
    "pages/letter_page.py",
    title="Surat Lamaran",
    icon="✉️",
)

assistant_page = st.Page(
    "pages/assistant_page.py",
    title="Asisten Karier",
    icon="💬",
)

navigation = st.navigation(
    [home_page, job_search_page, cv_match_page, skill_gap_page, letter_page, assistant_page],
    position="hidden",
)
navigation.run()
