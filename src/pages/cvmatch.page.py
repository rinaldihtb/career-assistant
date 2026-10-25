"""Streamlit page: Cocokkan CV (upload CV, koreksi profil, 10 lowongan paling cocok)."""

from src.components.cv_match import render_cv_match_page
from src.components.navigation import render_navigation

render_navigation()
render_cv_match_page()
