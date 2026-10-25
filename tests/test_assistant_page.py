"""Halaman Asisten Karier dengan Streamlit AppTest (tanpa browser dan tanpa API key).

``run_supervisor`` diganti fungsi palsu: yang dites di sini adalah UI, yaitu
riwayat chat, kartu lowongan, tabel SQL, dan penanganan error. Logika agent
sudah dites di file test masing-masing.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.components import assistant_chat
from src.components.assistant_chat import (
    DEPENDENCY_ERROR,
    EXAMPLE_QUESTIONS,
    MESSAGES_KEY,
    AssistantDependencies,
    prepare_display,
    split_rag_answer,
)
from src.models.agent import AgentResult, LLMCall, LLMUsage

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"
DEPS = AssistantDependencies(llm="llm", vector_store="store", db_path="db.sqlite")

RAG_RESULT = AgentResult(
    answer="Ada satu:\n- [ID:14] Data Analyst - PT Alpha (Jakarta): cocok.",
    source="rag",
    job_ids=[14],
    rows=[
        {
            "id": 14,
            "job_title": "Data Analyst",
            "company_name": "PT Alpha",
            "location": "Jakarta Selatan, Jakarta Raya\n(Hibrid)",
            "work_type": "Full time",
            "salary_min": 8_000_000,
            "salary_max": 10_000_000,
            "job_description": "Analisis data penjualan.",
        }
    ],
    usage=LLMUsage(
        llm_calls=2,
        total_tokens=2400,
        calls=[
            LLMCall(step="planner", model="gpt-4o-mini-2024-07-18", input_tokens=550,
                    output_tokens=30, total_tokens=580),
            LLMCall(step="rag_answer", model="gpt-4o-mini-2024-07-18", input_tokens=1600,
                    output_tokens=220, total_tokens=1820),
        ],
    ),
)  # fmt: skip
SQL_RESULT = AgentResult(
    answer="Rata-rata gaji Rp4.175.000 dari 4 lowongan.",
    source="sql",
    sql="SELECT AVG(salary_min) AS rata, COUNT(salary_min) AS n FROM jobs LIMIT 1",
    rows=[{"rata": 4_175_000, "n": 4}],
    usage=LLMUsage(llm_calls=3, total_tokens=1500),
)


def _page():
    from src.components.assistant_chat import render_assistant_page

    render_assistant_page()


@pytest.fixture
def calls(monkeypatch):
    """Ganti dependency dan supervisor; kembalikan daftar argumen setiap panggilan."""
    recorded: list[dict] = []
    replies = iter([RAG_RESULT, SQL_RESULT])

    def fake_run_supervisor(question, **kwargs):
        recorded.append({"question": question, **kwargs})
        return next(replies)

    monkeypatch.setattr(assistant_chat, "load_dependencies", lambda: DEPS)
    monkeypatch.setattr(assistant_chat, "run_supervisor", fake_run_supervisor)
    return recorded


def _app() -> AppTest:
    app = AppTest.from_function(_page, default_timeout=10)
    app.run()
    return app


def test_empty_chat_shows_example_questions(calls):
    app = _app()

    assert not app.exception
    assert [button.label for button in app.button] == list(EXAMPLE_QUESTIONS)
    assert len(app.chat_message) == 0
    assert calls == []


def test_rag_answer_shows_uniform_card_with_reason(calls):
    app = _app()
    app.chat_input[0].set_value("Lowongan data analyst di Jakarta").run()

    assert not app.exception
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
    texts = [md.value for md in app.markdown]
    assert "Ada satu:" in texts
    assert "**Data Analyst**  \nPT Alpha" in texts
    assert "**Kenapa cocok:** cocok." in texts
    assert not any("[ID:14]" in text for text in texts)  # daftar tidak tampil dua kali
    assert (
        "Lokasi: Jakarta Selatan, Jakarta Raya (Hibrid) | Tipe: Full time | "
        "Gaji: Rp8.000.000 - Rp10.000.000 per bulan | ID 14"
    ) in [caption.value for caption in app.caption]
    assert calls[0]["history"] == []
    assert calls[0]["db_path"] == "db.sqlite"


def test_llm_usage_detail_per_call(calls):
    app = _app()
    app.chat_input[0].set_value("Lowongan data analyst di Jakarta").run()

    assert app.expander[-1].label.startswith(
        "Detail: Pencarian lowongan (RAG) | 2 panggilan LLM | 2.400 token | "
    )
    usage_table = app.dataframe[-1].value.to_dict("records")
    assert [row["Langkah"] for row in usage_table] == [
        "Planner: pecah pertanyaan",
        "RAG: jawab dari kandidat",
    ]
    assert usage_table[1] == {
        "Langkah": "RAG: jawab dari kandidat",
        "Model": "gpt-4o-mini-2024-07-18",
        "Input": 1600,
        "Output": 220,
        "Total": 1820,
    }


def test_rag_answer_without_id_lines_falls_back_to_full_text(monkeypatch):
    result = RAG_RESULT.model_copy(update={"answer": "Lowongan ID 14 cocok untuk Anda."})
    monkeypatch.setattr(assistant_chat, "load_dependencies", lambda: DEPS)
    monkeypatch.setattr(assistant_chat, "run_supervisor", lambda question, **kwargs: result)
    app = _app()
    app.chat_input[0].set_value("Lowongan data analyst").run()

    texts = [md.value for md in app.markdown]
    assert "Lowongan ID 14 cocok untuk Anda." in texts
    assert not any(text.startswith("**Kenapa cocok:**") for text in texts)
    assert "**Data Analyst**  \nPT Alpha" in texts  # kartu tetap tampil


def test_second_question_sends_history_and_sql_shows_table(calls):
    app = _app()
    app.chat_input[0].set_value("Lowongan data analyst di Jakarta").run()
    app.chat_input[0].set_value("Rata-rata gajinya berapa?").run()

    assert not app.exception
    assert calls[1]["history"] == [
        {"role": "user", "content": "Lowongan data analyst di Jakarta"},
        {"role": "assistant", "content": RAG_RESULT.answer},
    ]
    assert len(app.chat_message) == 4
    tables = [table.value.to_dict("records") for table in app.dataframe]
    assert [{"rata": 4_175_000, "n": 4}] in tables
    assert app.code[0].value == SQL_RESULT.sql


def test_history_survives_rerun(calls):
    app = _app()
    app.chat_input[0].set_value("Lowongan data analyst di Jakarta").run()
    app.run()

    assert len(app.chat_message) == 2
    assert any("ID 14" in caption.value for caption in app.caption)
    assert len(calls) == 1  # rerun tidak memanggil supervisor lagi


def test_example_button_asks_question(calls):
    app = _app()
    app.button[0].click().run()

    assert calls[0]["question"] == EXAMPLE_QUESTIONS[0]
    assert len(app.chat_message) == 2


def test_new_conversation_clears_history(calls):
    app = _app()
    app.chat_input[0].set_value("Lowongan data analyst di Jakarta").run()
    app.button[0].click().run()  # "Mulai percakapan baru"

    assert app.session_state[MESSAGES_KEY] == []
    assert len(app.chat_message) == 0


def test_missing_config_shows_error_instead_of_crash(monkeypatch):
    def broken():
        raise KeyError("QDRANT_URL")

    monkeypatch.setattr(assistant_chat, "load_dependencies", broken)
    app = _app()

    assert not app.exception
    assert app.error[0].value == DEPENDENCY_ERROR
    assert len(app.chat_input) == 0


def test_page_is_registered_in_app(calls):
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.run()
    app.switch_page("pages/assistant.page.py").run()

    assert not app.exception
    assert app.markdown[0].value == "## Asisten Karier"


# --- Helper tanpa Streamlit --------------------------------------------------


def test_split_rag_answer_handles_nested_location():
    answer = (
        "Berikut lowongannya:\n\n"
        "- [ID:14] Data Analyst - PT A (Jakarta Selatan, Jakarta Raya (Hibrid)): Fokus SQL.\n"
        "- [ID:28] Consultant - PT B (Jakarta Raya): Kontrak, analitik."
    )
    assert split_rag_answer(answer) == (
        "Berikut lowongannya:",
        {14: "Fokus SQL.", 28: "Kontrak, analitik."},
    )


def test_split_rag_answer_without_location_separator():
    assert split_rag_answer("- [ID:7] Analyst tanpa format lokasi")[1] == {
        7: "Analyst tanpa format lokasi"
    }


SQL_LIST_RESULT = AgentResult(
    answer=(
        "Lowongan dengan gaji tertinggi:\n"
        "- [ID:4] Marketing Manager - PT Delta (Jakarta Raya): Gaji maksimum Rp20.000.000."
    ),
    source="sql",
    job_ids=[4, 1],
    sql="SELECT id, job_title, salary_max FROM jobs ORDER BY salary_max DESC LIMIT 2",
    rows=[{"id": 4, "salary_max": 20_000_000}, {"id": 1, "salary_max": 10_000_000}],
)


def test_prepare_display_sql_job_lines_become_cards(jobs_db):
    display = prepare_display(SQL_LIST_RESULT, jobs_db)

    assert display.text == "Lowongan dengan gaji tertinggi:"
    assert [(job["id"], note) for job, note in display.cards] == [
        (4, "Gaji maksimum Rp20.000.000.")
    ]
    assert display.cards[0][0]["job_title"] == "Marketing Manager"  # detail dari SQLite


def test_prepare_display_sql_aggregate_has_no_cards(jobs_db):
    display = prepare_display(SQL_RESULT, jobs_db)

    assert display.text == SQL_RESULT.answer
    assert display.cards == []


def test_prepare_display_sql_database_error_falls_back_to_text(tmp_path):
    display = prepare_display(SQL_LIST_RESULT, tmp_path / "tidak_ada.sqlite")

    assert display.text == SQL_LIST_RESULT.answer
    assert display.cards == []


def test_sql_job_list_renders_cards_and_table(monkeypatch, jobs_db):
    deps = AssistantDependencies(llm="llm", vector_store="store", db_path=str(jobs_db))
    monkeypatch.setattr(assistant_chat, "load_dependencies", lambda: deps)
    monkeypatch.setattr(assistant_chat, "run_supervisor", lambda q, **kwargs: SQL_LIST_RESULT)
    app = _app()
    app.chat_input[0].set_value("Lowongan dengan gaji tertinggi?").run()

    texts = [md.value for md in app.markdown]
    assert "**Keterangan:** Gaji maksimum Rp20.000.000." in texts
    assert "**Marketing Manager**  \nPT Delta" in texts
    assert [{"id": 4, "salary_max": 20_000_000}, {"id": 1, "salary_max": 10_000_000}] in [
        table.value.to_dict("records") for table in app.dataframe
    ]
    assert app.expander[-1].label.startswith("Detail: Statistik data (SQL)")


def test_dollar_sign_in_answer_is_not_latex(monkeypatch):
    result = AgentResult(answer="Gaji $1000 sampai $2000.", source="general")
    monkeypatch.setattr(assistant_chat, "load_dependencies", lambda: DEPS)
    monkeypatch.setattr(assistant_chat, "run_supervisor", lambda q, **kwargs: result)
    app = _app()
    app.chat_input[0].set_value("Halo").run()

    assert r"Gaji \$1000 sampai \$2000." in [md.value for md in app.markdown]


# --- Jawaban gabungan beberapa agent -----------------------------------------

MULTI_RESULT = AgentResult(
    answer="(gabungan, dipakai untuk riwayat)",
    source="multi",
    job_ids=[14],
    parts=[
        SQL_RESULT.model_copy(update={"answer": f"**Rata-rata gaji?**\n{SQL_RESULT.answer}"}),
        RAG_RESULT.model_copy(update={"answer": f"**Lowongan data analyst**\n{RAG_RESULT.answer}"}),
        AgentResult(answer="Catatan: sebagian permintaan tidak dijalankan.", source="general"),
    ],
    usage=LLMUsage(llm_calls=4, total_tokens=4000),
)


def test_prepare_display_multi_builds_display_per_part(jobs_db):
    display = prepare_display(MULTI_RESULT, jobs_db)

    assert display.text == "" and display.cards == []
    assert [len(part.cards) for part in display.parts] == [0, 1, 0]
    assert display.parts[1].text == "**Lowongan data analyst**\nAda satu:"


def test_multi_answer_renders_every_part_and_one_usage_panel(monkeypatch):
    monkeypatch.setattr(assistant_chat, "load_dependencies", lambda: DEPS)
    monkeypatch.setattr(assistant_chat, "run_supervisor", lambda q, **kwargs: MULTI_RESULT)
    app = _app()
    app.chat_input[0].set_value("Rata-rata gaji dan lowongan data analyst").run()

    assert not app.exception
    texts = [md.value for md in app.markdown]
    assert f"**Rata-rata gaji?**\n{SQL_RESULT.answer}" in texts
    assert "**Lowongan data analyst**\nAda satu:" in texts
    assert "**Data Analyst**  \nPT Alpha" in texts  # kartu RAG
    assert "**Kenapa cocok:** cocok." in texts
    assert "Catatan: sebagian permintaan tidak dijalankan." in texts
    assert [{"rata": 4_175_000, "n": 4}] in [t.value.to_dict("records") for t in app.dataframe]
    usage_panels = [e.label for e in app.expander if e.label.startswith("Detail:")]
    assert len(usage_panels) == 1
    assert usage_panels[0].startswith("Detail: Gabungan beberapa agent | 4 panggilan LLM")
