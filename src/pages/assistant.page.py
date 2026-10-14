"""Streamlit page: Asisten Karier (chat dengan supervisor agent)."""

from components.assistant_chat import render_assistant_page
from components.navigation import render_navigation

render_navigation()
render_assistant_page()
