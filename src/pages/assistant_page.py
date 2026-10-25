"""Streamlit page: Asisten Karier (chat dengan supervisor agent)."""

from src.components.assistant_chat import render_assistant_page
from src.components.navigation import render_navigation

render_navigation()
render_assistant_page()
