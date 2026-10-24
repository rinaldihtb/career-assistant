"""Halaman Streamlit: Cari Lowongan (daftar dan filter lowongan)."""

from components.job_search import render_job_search_page
from components.navigation import render_navigation

render_navigation()
render_job_search_page()
