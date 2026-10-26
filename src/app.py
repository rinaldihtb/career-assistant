"""Streamlit application entry point."""

import streamlit as st

st.set_page_config(
    page_title="Career Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="locked",
)

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
