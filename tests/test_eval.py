"""Set eval dan pemeriksaannya, tanpa API key.

- Skema tests/eval/queries.yaml dan tests/eval/heldout.yaml valid dan gold
  SQL-nya jalan di data asli (dilewati jika data/database.sqlite belum di-seed,
  atau jika heldout.yaml belum ditulis). Pesan gagal hanya menampilkan ID kasus,
  supaya isi held-out tidak terbaca saat ``make test``.
- Fungsi pemeriksaan di ``evaluation.checks`` dites dengan contoh kecil.
- ``run_case`` dites ujung ke ujung dengan LLM palsu dan database fixture.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from conftest import RecordingFakeChatModel
from langchain_core.messages import AIMessage

from agents.prompts import SUPERVISOR_GENERAL_MESSAGE
from agents.supervisor_agent import SupervisorContext
from evaluation.checks import (
    RunOutput,
    check_rag,
    check_sql_ids,
    check_sql_values,
    extract_numbers,
    extract_run,
    llm_call_budget,
    number_present,
    route_ok,
    run_checks,
    split_citations,
)
from evaluation.run_eval import load_cases, rescore_record, run_case, summarize
from models.agent import AgentResult

CASES_PATH = Path(__file__).parent / "eval" / "queries.yaml"
HELDOUT_PATH = Path(__file__).parent / "eval" / "heldout.yaml"
CASE_SETS = [
    pytest.param(CASES_PATH, id="queries"),
    pytest.param(
        HELDOUT_PATH,
        id="heldout",
        marks=pytest.mark.skipif(not HELDOUT_PATH.exists(), reason="heldout.yaml belum ditulis"),
    ),
]
REAL_DB = Path("data/database.sqlite")
CHECK_KEYS = {
    "sql_values",
    "sql_ids",
    "rag",
    "no_ids",
    "answer_any",
    "answer_none",
    "no_rupiah",
    "standalone_any",
    "standalone_none",
}
RESPONSES = {"answer", "clarify", "refuse", "out_of_scope", "unsupported"}
USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


# --- Set eval ----------------------------------------------------------------


@pytest.mark.parametrize("cases_path", CASE_SETS)
def test_cases_schema(cases_path):
    cases = load_cases(cases_path)
    ids = [case["id"] for case in cases]

    assert len(ids) == len(set(ids))
    for case in cases:
        expected = case["expected"]
        assert case["question"].strip(), case["id"]
        assert expected["response"] in RESPONSES, case["id"]
        assert set(expected["agents"]) <= {"sql", "rag"} and len(expected["agents"]) <= 3
        if expected["response"] != "answer":
            assert expected["agents"] == [], case["id"]
        assert set(case.get("checks", {})) <= CHECK_KEYS, case["id"]
        for group in case.get("checks", {}).get("rag", []):
            assert ("title_any" in group) != ("desc_any" in group), case["id"]
        for message in case.get("history", []):
            assert message["role"] in {"user", "assistant"} and message["content"], case["id"]


def test_load_cases_filters_by_prefix():
    ids = [case["id"] for case in load_cases(CASES_PATH, ["T3", "T6b-01"])]
    assert ids and all(i.startswith("T3") or i == "T6b-01" for i in ids)


@pytest.mark.skipif(not HELDOUT_PATH.exists(), reason="heldout.yaml belum ditulis")
def test_heldout_is_separate_from_queries():
    """Held-out memakai ID berawalan "H" dan tidak menyalin pertanyaan queries.yaml.

    ID berawalan "H" membuat hasil held-out tidak tercampur dengan kasus T* di
    laporan per kasus; kategori tetap T1a..T13 supaya ringkasan per kategori
    bisa dibandingkan langsung dengan queries.yaml.
    """
    known = {" ".join(case["question"].lower().split()) for case in load_cases(CASES_PATH)}
    heldout = load_cases(HELDOUT_PATH)

    assert [case["id"] for case in heldout if not case["id"].startswith("H")] == []
    copied = [case["id"] for case in heldout if " ".join(case["question"].lower().split()) in known]
    assert copied == []


@pytest.mark.skipif(not REAL_DB.exists(), reason="data/database.sqlite belum di-seed")
@pytest.mark.parametrize("cases_path", CASE_SETS)
def test_gold_sql_runs_on_real_database(cases_path):
    uri = f"file:{REAL_DB}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        for case in load_cases(cases_path):
            checks = case.get("checks", {})
            for sql in checks.get("sql_values", []):
                assert conn.execute(sql).fetchall(), case["id"]
            for sql in checks.get("sql_ids", []):
                ids = [row[0] for row in conn.execute(sql)]
                assert 0 < len(ids) <= 10, (case["id"], ids)  # muat di MAX_CARDS UI


# --- Angka dan sitasi --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rata-rata Rp6.812.500.", [6_812_500]),
        ("sekitar Rp6,8 juta", [6_800_000]),
        ("15jt", [15_000_000]),
        ("4 dari 16 lowongan", [4, 16]),
        ("- [ID:473] Investment Associate", []),
        ("total 10.000", [10_000]),
    ],
)
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == expected


def test_number_present_tolerance():
    assert number_present(6_812_500, [6_800_000])  # pembulatan wajar
    assert not number_present(6_812_500, [6_000_000])
    assert number_present(16, [16.0])
    assert not number_present(16, [17])  # angka kecil harus persis


def test_split_citations_by_agent_and_invalid():
    run = RunOutput(
        answer="[ID:1] a\n[ID:2] b\n[ID:9] c",
        subresults=[
            AgentResult(answer="", source="sql", job_ids=[1, 3]),
            AgentResult(answer="", source="rag", job_ids=[2]),
        ],
    )
    assert split_citations(run) == ([1], [2], [9])


# --- Pemeriksaan -------------------------------------------------------------


def test_check_sql_values():
    gold = [(6812500.0, 8625000.0, 4, 16)]
    ok = "Min Rp6.812.500, maks Rp8.625.000 (4 dari 16 lowongan bergaji)."
    assert check_sql_values(gold, ok).passed
    result = check_sql_values(gold, "Rata-rata Rp6.812.500.")
    assert not result.passed and "8625000" in result.detail


def test_check_sql_values_ignores_text_and_null():
    assert check_sql_values([("Full time", 407, None)], "Full time: 407").passed


def test_check_sql_ids():
    assert check_sql_ids([10, 14], [14, 10]).passed
    result = check_sql_ids([10, 14], [10, 99])
    assert not result.passed and result.detail == "kurang [14], lebih [99]"


def test_check_rag_groups_and_off_topic():
    jobs = {
        1: {"job_title": "Graphic Designer", "location": "Jakarta Raya"},
        2: {"job_title": "Social Media Staff", "location": "Jakarta Barat"},
        3: {"job_title": "Sales", "location": "Jakarta Raya"},
    }
    groups = [
        {"title_any": ["design"], "location_any": ["jakarta"], "min_ids": 1},
        {"title_any": ["social media"], "min_ids": 1},
    ]
    assert check_rag(groups, [1, 2], jobs).passed
    assert "grup 2" in check_rag(groups, [1], jobs).detail
    assert "tidak relevan: [3]" in check_rag(groups, [1, 2, 3], jobs).detail


def test_check_rag_location_none_rejects_excluded_city():
    jobs = {
        1: {"job_title": "Sales Executive", "location": "Jakarta Selatan, Jakarta Raya"},
        2: {"job_title": "Sales", "location": "Surabaya, Jawa Timur"},
    }
    groups = [{"title_any": ["sales"], "location_none": ["jakarta"], "min_ids": 1}]
    assert check_rag(groups, [2], jobs).passed
    assert "tidak relevan: [1]" in check_rag(groups, [1, 2], jobs).detail
    assert "grup 1: 0 cocok" in check_rag(groups, [1], jobs).detail


def test_check_rag_by_description():
    jobs = {5: {"job_title": "Backend", "job_description": "Python dan SQL", "location": "-"}}
    assert check_rag([{"desc_any": ["python"], "min_ids": 1}], [5], jobs).passed


def test_run_checks_text_rules():
    case = {
        "checks": {
            "no_ids": True,
            "no_rupiah": True,
            "answer_any": ["posisi"],
            "standalone_none": ["jakarta"],
        }
    }
    run = RunOutput(answer="Sebutkan posisi atau kota.", standalone=["Lowongan di Surabaya"])
    assert all(check.passed for check in run_checks(case, run, {}, {}))

    bad = RunOutput(answer="[ID:1] gajinya Rp5.000.000", standalone=["di Jakarta"])
    failed = {check.name for check in run_checks(case, bad, {}, {}) if not check.passed}
    assert failed == {"no_ids", "no_rupiah", "answer_any", "standalone_none"}


def test_route_ok_ignores_order_and_accepts_alternative():
    expected = {"agents": ["sql", "rag"], "agents_alt": [["sql"]]}
    assert route_ok(expected, ["rag", "sql"])
    assert route_ok(expected, ["sql"])
    assert not route_ok(expected, ["rag"])


def test_llm_call_budget():
    assert llm_call_budget([]) == 1
    assert llm_call_budget(["sql"]) == 4
    assert llm_call_budget(["sql", "rag"]) == 6  # planner + 3 + 1 + sintesis


# --- Bentuk state supervisor -------------------------------------------------


def test_extract_run_single_route_state():
    result = AgentResult(answer="Ada 3.", source="sql", job_ids=[1])
    run = extract_run({"route": "sql", "standalone_question": "Berapa?", "result": result})
    assert run.agents == ["sql"] and run.standalone == ["Berapa?"] and run.subresults == [result]


def test_extract_run_general_route_has_no_agent():
    result = AgentResult(answer="Halo", source="general")
    run = extract_run({"route": "general", "standalone_question": "Halo", "result": result})
    assert run.agents == [] and run.subresults == []


def test_extract_run_multi_subtask_state():
    subs = [AgentResult(answer="a", source="sql"), AgentResult(answer="b", source="rag")]
    state = {
        "subtasks": [{"agent": "sql", "question": "q1"}, {"agent": "rag", "question": "q2"}],
        "subresults": subs,
        "result": AgentResult(answer="a\nb", source="sql"),
    }
    run = extract_run(state)
    assert run.agents == ["sql", "rag"] and run.standalone == ["q1", "q2"]
    assert run.subresults == subs and run.answer == "a\nb"


# --- run_case ujung ke ujung (LLM palsu) -------------------------------------


class FakeToolCallingLLM(RecordingFakeChatModel):
    """LLM palsu yang mendukung ``with_structured_output`` (pola test_supervisor.py)."""

    def bind_tools(self, tools, **kwargs):
        return self


def _llm(*replies):
    def _replies():
        for reply in replies:
            yield reply if isinstance(reply, AIMessage) else AIMessage(reply, usage_metadata=USAGE)

    return FakeToolCallingLLM(messages=_replies())


def _plan(response, *subtasks):
    args = {
        "response": response,
        "subtasks": [{"agent": agent, "question": question} for agent, question in subtasks],
        "skipped": [],
        "declined": [],
    }
    return AIMessage(
        content="", tool_calls=[{"name": "Plan", "args": args, "id": "1"}], usage_metadata=USAGE
    )


SQL_CASE = {
    "id": "X-01",
    "category": "X",
    "question": "Berapa lowongan di Jakarta?",
    "expected": {"response": "answer", "agents": ["sql"]},
    "checks": {"sql_values": ["SELECT COUNT(*) FROM jobs WHERE lower(location) LIKE '%jakarta%'"]},
}


def test_run_case_passes_correct_sql_answer(jobs_db, fake_vector_store):
    llm = _llm(
        _plan("answer", ("sql", "Berapa lowongan di Jakarta?")),
        "```sql\nSELECT COUNT(*) AS n FROM jobs WHERE lower(location) LIKE '%jakarta%'\n```",
        "Ada 3 lowongan di Jakarta.",
    )
    context = SupervisorContext(llm=llm, vector_store=fake_vector_store([]), db_path=jobs_db)
    record = run_case(SQL_CASE, context)

    assert record["passed"] and record["route_ok"]
    assert record["llm_calls"] == 3 and record["within_budget"]
    assert record["total_tokens"] == 45


def test_run_case_fails_wrong_number_and_route(jobs_db, fake_vector_store):
    llm = _llm(_plan("out_of_scope"))
    context = SupervisorContext(llm=llm, vector_store=fake_vector_store([]), db_path=jobs_db)
    record = run_case(SQL_CASE, context)

    assert record["answer"] == SUPERVISOR_GENERAL_MESSAGE
    assert not record["passed"] and not record["route_ok"]
    assert record["checks"][0]["detail"] == "tidak ada di jawaban: [3]"


def test_run_case_records_graph_exception(jobs_db, fake_vector_store):
    class BrokenGraph:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("graph rusak")

    context = SupervisorContext(llm=_llm(), vector_store=fake_vector_store([]), db_path=jobs_db)
    record = run_case(SQL_CASE, context, graph=BrokenGraph())

    assert not record["passed"]
    assert record["error"] == "RuntimeError: graph rusak"


def test_summarize_counts():
    records = [
        {
            "category": "T1a",
            "passed": True,
            "route_ok": True,
            "invalid_citations": [],
            "within_budget": True,
            "llm_calls": 3,
            "total_tokens": 100,
            "seconds": 2.0,
        },
        {
            "category": "T3",
            "passed": False,
            "route_ok": False,
            "invalid_citations": [9],
            "within_budget": True,
            "llm_calls": 3,
            "total_tokens": 200,
            "seconds": 4.0,
        },
    ]
    summary = summarize(records)
    assert summary["passed"] == 1 and summary["citations_valid"] == 1
    assert summary["avg_llm_calls"] == 3 and summary["total_tokens"] == 300
    assert summary["by_category"]["T3"] == {"cases": 1, "passed": 0, "route_ok": 0}


def test_rescore_record_reuses_answer_and_cost(jobs_db, fake_vector_store):
    llm = _llm(
        _plan("answer", ("sql", "Berapa lowongan di Jakarta?")),
        "```sql\nSELECT COUNT(*) AS n FROM jobs WHERE lower(location) LIKE '%jakarta%'\n```",
        "Ada 3 lowongan di Jakarta.",
    )
    context = SupervisorContext(llm=llm, vector_store=fake_vector_store([]), db_path=jobs_db)
    original = run_case(SQL_CASE, context)

    same = rescore_record(SQL_CASE, original, jobs_db)
    assert {k: same[k] for k in original} == original  # kasus sama: hasil identik

    stricter = {**SQL_CASE, "checks": {**SQL_CASE["checks"], "answer_any": ["Bandung"]}}
    rescored = rescore_record(stricter, original, jobs_db)
    assert not rescored["passed"]
    assert rescored["llm_calls"] == 3 and rescored["total_tokens"] == 45  # biaya tidak berubah


def test_rescore_keeps_crashed_case_failed(jobs_db):
    crashed = {
        "answer": "",
        "agents": [],
        "standalone": [],
        "cited_sql": [],
        "cited_rag": [],
        "llm_calls": 0,
        "total_tokens": 0,
        "models": [],
        "seconds": 0.1,
        "error": "RuntimeError: graph rusak",
    }
    case = {**SQL_CASE, "checks": {"no_ids": True}}
    assert not rescore_record(case, crashed, jobs_db)["passed"]
