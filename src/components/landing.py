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
            st.switch_page("pages/jobsearch.page.py")

    with right_column:
        with st.container(border=True):
            st.subheader("Satu tempat untuk pencarian karier")
            st.write(
                "Cari lowongan, baca detail pekerjaan, dan bandingkan kebutuhan "
                "role dengan profil Anda."
            )
            st.caption(
                "Dibangun untuk membantu pencari kerja membuat keputusan yang lebih jelas."
            )


def render_feature_cards() -> None:
    """Describe the core application capabilities."""
    st.header("Apa yang bisa Anda lakukan?")
    search_column, insight_column, plan_column = st.columns(3)

    with search_column:
        with st.container(border=True):
            st.subheader("🔎 Cari lowongan")
            st.write(
                "Gunakan kata kunci, lokasi, tipe kerja, dan rentang gaji untuk "
                "mempersempit pencarian."
            )

    with insight_column:
        with st.container(border=True):
            st.subheader("📄 Pahami kebutuhan role")
            st.write(
                "Baca detail tanggung jawab, kualifikasi, lokasi, dan informasi "
                "kompensasi dalam satu tampilan."
            )

    with plan_column:
        with st.container(border=True):
            st.subheader("✨ Siapkan karier Anda")
            st.write(
                "Fitur analisis CV dan rekomendasi skill dapat membantu Anda "
                "menyiapkan langkah berikutnya."
            )


def render_how_it_works() -> None:
    """Render a short, descriptive application flow."""
    st.header("Cara kerja")
    first, second, third = st.columns(3)
    first.metric("1", "Cari role yang diminati")
    second.metric("2", "Pelajari detail lowongan")
    third.metric("3", "Tentukan langkah persiapan")


def render_landing_page() -> None:
    """Compose a descriptive landing page without job-search content."""
    st.markdown("# Career Assistant")
    render_hero()
    # st.divider()
    # render_feature_cards()
    # st.divider()
    # render_how_it_works()
