"""Streamlit page for browsing and filtering job vacancies."""

from src.components.job_search import render_job_search_page
from src.components.navigation import render_navigation

render_navigation()
render_job_search_page()
