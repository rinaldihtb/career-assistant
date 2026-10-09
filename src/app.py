"""Streamlit application entry point."""

import streamlit as st

st.set_page_config(
    page_title="Career Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="locked",
)

home_page = st.Page(
    "pages/home.page.py",
    title="Beranda",
    icon="🏠",
    default=True,
)
job_search_page = st.Page(
    "pages/jobsearch.page.py",
    title="Cari Lowongan",
    icon="🔎",
)

navigation = st.navigation([home_page, job_search_page], position="hidden")
navigation.run()
