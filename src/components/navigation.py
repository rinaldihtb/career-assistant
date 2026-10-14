"""Sidebar navigation for the Career Assistant pages."""

import streamlit as st


def render_navigation() -> None:
    """Render the shared sidebar navigation."""
    with st.sidebar:
        # st.title("App Name")
        # st.caption("App Description")
        # st.divider()
        st.page_link("pages/home.page.py", label="Home")
        st.page_link("pages/jobsearch.page.py", label="Search Job")
        st.page_link("pages/assistant.page.py", label="Career Assistant")
        st.page_link("pages/jobsearch.page.py", label="Sign in / Sign up")
