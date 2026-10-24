"""Halaman Streamlit: Surat lamaran untuk lowongan yang dipilih di Cocokkan CV."""

from components.cover_letter import render_cover_letter_page
from components.navigation import render_navigation

render_navigation()
render_cover_letter_page()
