"""Streamlit page for browsing and filtering job vacancies."""

from components.job_search import render_job_search_page
from components.navigation import render_navigation

render_navigation()
render_job_search_page()
