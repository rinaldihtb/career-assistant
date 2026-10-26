"""Sidebar navigation for the Career Assistant pages."""

import streamlit as st

from src.cv.ingest import CV_STATE_KEY

CV_PAGE = "pages/cvmatch_page.py"


def render_navigation() -> None:
    """Render the shared sidebar navigation and the active-CV status block.

    Tanpa akun (keputusan produk): tidak ada link Sign in / Sign up. Status CV dibaca dari
    ``st.session_state`` saja, jadi tampil sama di semua halaman.
    """
    with st.sidebar:
        # st.title("App Name")
        # st.caption("App Description")
        # st.divider()
        st.page_link("pages/home_page.py", label="Beranda")
        st.page_link("pages/jobsearch_page.py", label="Cari Lowongan")
        st.page_link(CV_PAGE, label="Cocokkan CV")
        st.page_link("pages/skillgap_page.py", label="Skill Gap")
        st.page_link("pages/assistant_page.py", label="Asisten Karier")
        st.divider()
        # Status saja, bukan link kedua: link "Cocokkan CV" di atas sudah ke halaman yang sama,
        # dan dua link aktif ke satu halaman tampil ganda di sidebar.
        cv = st.session_state.get(CV_STATE_KEY)
        st.caption(f"CV aktif: {cv['file_name']}" if cv else "Belum ada CV")
