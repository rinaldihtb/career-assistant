"""Reusable sections for the descriptive Career Assistant landing page."""

import streamlit as st


def render_hero() -> None:
    """Render the introductory section and primary call to action."""
    left_column, right_column = st.columns([3, 2], gap="large")

    with left_column:
        st.markdown("### 💼 Career Assistant")
        st.title("Temukan langkah karier Anda berikutnya.")
        st.write(
            "Career Assistant membantu Anda menemukan lowongan relevan, "
            "memahami kebutuhan role, dan mempersiapkan langkah karier secara lebih terarah."
        )
        if st.button("Jelajahi lowongan", type="primary"):
            st.switch_page("pages/jobsearch_page.py")

    with right_column, st.container(border=True):
        st.subheader("Satu tempat untuk pencarian karier")
        st.write(
            "Cari lowongan, baca detail pekerjaan, dan bandingkan kebutuhan "
            "role dengan profil Anda."
        )
        st.caption("Dibangun untuk membantu pencari kerja membuat keputusan yang lebih jelas.")


def render_landing_page() -> None:
    """Compose a descriptive landing page without job-search content."""
    st.markdown("# Career Assistant")
    render_hero()
