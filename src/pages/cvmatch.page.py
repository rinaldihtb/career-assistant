"""Streamlit page: Cocokkan CV (upload CV, koreksi profil, 10 lowongan paling cocok)."""

from components.cv_match import render_cv_match_page
from components.navigation import render_navigation

render_navigation()
render_cv_match_page()
