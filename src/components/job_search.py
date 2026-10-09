import streamlit as st


def render_search_form():
    locations = ["Semua lokasi"]
    work_types = ["Semua tipe kerja"]
    salary_options = {
        "Semua rentang gaji": 0,
        "Minimal Rp5 juta": 5_000_000,
        "Minimal Rp10 juta": 10_000_000,
        "Minimal Rp15 juta": 15_000_000,
    }
    salary_labels = list(salary_options)
    st.session_state.setdefault("location", locations[0])
    st.session_state.setdefault("work_type", work_types[0])
    st.session_state.setdefault("minimum_salary", 0)
    current_salary_label = next(
        (
            label
            for label, value in salary_options.items()
            if value == st.session_state.minimum_salary
        ),
        salary_labels[0],
    )

    with st.form("job-search"):
        input_column, button_column = st.columns([5, 1])
        keyword = input_column.text_input(
            "Search Job",
            value=st.session_state.get("keyword", ""),
            placeholder="Search Position",
            label_visibility="collapsed",
        )
        submitted = button_column.form_submit_button(
            "Search", type="primary", use_container_width=True
        )

        # Filter berada pada satu baris, di bawah pencarian dan tombol submit.
        location_column, type_column, salary_column = st.columns(3)
        location = location_column.selectbox(
            "Lokasi",
            locations,
            index=locations.index(st.session_state.location)
            if st.session_state.location in locations
            else 0,
        )
        work_type = type_column.selectbox(
            "Tipe kerja",
            work_types,
            index=work_types.index(st.session_state.work_type)
            if st.session_state.work_type in work_types
            else 0,
        )
        salary_label = salary_column.selectbox(
            "Gaji", salary_labels, index=salary_labels.index(current_salary_label)
        )

    if submitted:
        st.session_state.keyword = keyword
        st.session_state.location = location
        st.session_state.work_type = work_type
        st.session_state.minimum_salary = salary_options[salary_label]


def render_job_list():
    st.markdown("## List Job")


def render_job_detail():
    st.markdown("## Detail Job")


def render_content():
    list_column, detail_column = st.columns([3, 7], gap="small")

    with list_column:
        render_job_list()

    with detail_column:
        render_job_detail()


def render_job_search_page():
    st.markdown("## Search Job")
    render_search_form()

    render_content()
