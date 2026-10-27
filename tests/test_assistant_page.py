"""Halaman Asisten Karier dengan Streamlit AppTest (tanpa browser dan tanpa API key).

``get_supervisor_agent`` diganti graph palsu, dan ``requests.post`` dari UI diteruskan ke
handler ``POST /chat-assistant`` yang asli, sehingga kontrak UI <-> API ikut teruji tanpa
server. Yang dites di sini adalah UI, yaitu riwayat chat, kartu lowongan, tabel SQL, dan
penanganan error. Logika agent sudah dites di file test masing-masing.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.components import assistant_chat
from src.components.assistant_chat import (
    API_ERROR,
    DEPENDENCY_ERROR,
    EXAMPLE_QUESTIONS,
    MESSAGES_KEY,
    prepare_display,
    split_rag_answer,
)
from src.models.agent import AgentResult, LLMCall, LLMUsage
from src.routes import routes

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"
# Run pertama di satu proses pytest memuat pandas/pyarrow untuk st.dataframe: 7 detik di
# workspace dan lebih dari 10 detik di laptop macOS. Run berikutnya di bawah 1 detik.
APP_TIMEOUT = 30

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


class FakeGraph:
    """Pengganti ``SupervisorAgent.graph``: mencatat input, membalas ``AgentResult`` berurutan.

    Hasil terakhir dipakai ulang jika pertanyaan lebih banyak dari daftar hasil. ``urls``
    diisi ``_use`` dengan alamat setiap panggilan API dari UI.
    """

    def __init__(self, *results: AgentResult) -> None:
        self.inputs: list[dict] = []
        self.urls: list[str] = []
        self.results = results

    def invoke(self, state: dict) -> dict:
        self.inputs.append(state)
        return {"result": self.results[min(len(self.inputs), len(self.results)) - 1]}


class FakeResponse:
    """Respons ``requests`` minimal: status 200 dan isi JSON dari handler route."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def _use(monkeypatch, *results: AgentResult, repository=None) -> FakeGraph:
    """Pasang graph palsu di UI dan route, lalu arahkan ``requests.post`` ke handler route."""
    graph = FakeGraph(*results)
    agent = SimpleNamespace(graph=graph)
    monkeypatch.setattr(assistant_chat, "get_supervisor_agent", lambda: agent)
    monkeypatch.setattr(routes, "get_supervisor_agent", lambda: agent)

    def post(url, *, json, timeout):
        graph.urls.append(url)
        return FakeResponse(routes.chat_assistant(json))

    monkeypatch.setattr(assistant_chat.requests, "post", post)
    if repository is not None:
        monkeypatch.setattr(assistant_chat, "get_job_repository", lambda: repository)
    return graph


@pytest.fixture
def calls(monkeypatch):
    """Graph palsu RAG lalu SQL; kembalikan daftar input ``invoke`` setiap pertanyaan."""
    return _use(monkeypatch, RAG_RESULT, SQL_RESULT).inputs


def _app() -> AppTest:
    app = AppTest.from_function(_page, default_timeout=APP_TIMEOUT)
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
    assert calls[0] == {
        "messages": [{"role": "user", "content": "Lowongan data analyst di Jakarta"}]
    }


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
    _use(monkeypatch, result)
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
    assert calls[1]["messages"] == [
        {"role": "user", "content": "Lowongan data analyst di Jakarta"},
        {"role": "assistant", "content": RAG_RESULT.answer},
        {"role": "user", "content": "Rata-rata gajinya berapa?"},
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

    assert calls[0]["messages"][-1]["content"] == EXAMPLE_QUESTIONS[0]
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

    monkeypatch.setattr(assistant_chat, "get_supervisor_agent", broken)
    app = _app()

    assert not app.exception
    assert app.error[0].value == DEPENDENCY_ERROR
    assert len(app.chat_input) == 0


def test_api_url_comes_from_env(monkeypatch):
    graph = _use(monkeypatch, SQL_RESULT)
    monkeypatch.setenv("ASSISTANT_API_URL", "http://fastapi:8000/")
    app = _app()
    app.chat_input[0].set_value("Rata-rata gaji di Surabaya?").run()

    monkeypatch.delenv("ASSISTANT_API_URL")
    app.chat_input[0].set_value("Kalau di Bandung?").run()

    assert graph.urls == [
        "http://fastapi:8000/chat-assistant",
        "http://localhost:8000/chat-assistant",
    ]


def test_unreachable_api_shows_error_and_keeps_question(monkeypatch):
    _use(monkeypatch, SQL_RESULT)

    def down(url, *, json, timeout):
        raise requests.ConnectionError("tidak terjangkau")

    monkeypatch.setattr(assistant_chat.requests, "post", down)
    app = _app()
    app.chat_input[0].set_value("Rata-rata gaji di Surabaya?").run()

    assert not app.exception
    assert app.error[0].value == API_ERROR
    assert [m["role"] for m in app.session_state[MESSAGES_KEY]] == ["user"]


def test_page_is_registered_in_app(calls):
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    app.run()
    app.switch_page("pages/assistant_page.py").run()

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


@pytest.fixture
def repository(monkeypatch, make_repository):
    """``get_job_repository`` halaman Asisten diarahkan ke database fixture."""
    repository = make_repository()
    monkeypatch.setattr(assistant_chat, "get_job_repository", lambda: repository)
    return repository


def test_prepare_display_sql_job_lines_become_cards(repository):
    display = prepare_display(SQL_LIST_RESULT)

    assert display.text == "Lowongan dengan gaji tertinggi:"
    assert [(job["id"], note) for job, note in display.cards] == [
        (4, "Gaji maksimum Rp20.000.000.")
    ]
    assert display.cards[0][0]["job_title"] == "Marketing Manager"  # detail dari SQLite


def test_prepare_display_sql_aggregate_has_no_cards(repository):
    display = prepare_display(SQL_RESULT)

    assert display.text == SQL_RESULT.answer
    assert display.cards == []


def test_prepare_display_sql_database_error_falls_back_to_text(monkeypatch):
    def broken_lookup(job_ids):
        raise RuntimeError("database tidak bisa dibuka")

    broken = SimpleNamespace(get_jobs_by_ids=broken_lookup)
    monkeypatch.setattr(assistant_chat, "get_job_repository", lambda: broken)
    display = prepare_display(SQL_LIST_RESULT)

    assert display.text == SQL_LIST_RESULT.answer
    assert display.cards == []


def test_sql_job_list_renders_cards_and_table(monkeypatch, make_repository):
    _use(monkeypatch, SQL_LIST_RESULT, repository=make_repository())
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
    _use(monkeypatch, result)
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


def test_prepare_display_multi_builds_display_per_part(repository):
    display = prepare_display(MULTI_RESULT)

    assert display.text == "" and display.cards == []
    assert [len(part.cards) for part in display.parts] == [0, 1, 0]
    assert display.parts[1].text == "**Lowongan data analyst**\nAda satu:"


def test_multi_answer_renders_every_part_and_one_usage_panel(monkeypatch):
    _use(monkeypatch, MULTI_RESULT)
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
