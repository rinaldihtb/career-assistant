"""Halaman Streamlit: Surat lamaran untuk lowongan yang dipilih di Cocokkan CV."""

from src.components.cover_letter import render_cover_letter_page
from src.components.navigation import render_navigation

render_navigation()
render_cover_letter_page()
