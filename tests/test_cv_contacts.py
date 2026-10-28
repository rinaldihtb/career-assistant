"""Kontak dari teks CV (regex, tanpa LLM) dan jaminan bahwa data CV tidak dikirim keluar.

Kontak tidak masuk prompt model; nama, kontak, dan profil tidak masuk span Langfuse.
"""

import json
from functools import partial
from pathlib import Path

import langfuse
import pytest
from conftest import ScriptedLLM, agent_script, plan_reply
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.agents.rag_agent import RagAgent
from src.agents.sql_agent import SqlAgent
from src.agents.supervisor_agent_graph import SupervisorAgent, planner_messages
from src.containers import traced_llm
from src.cv.contacts import extract_contacts
from src.cv.gap import compute_skill_gap
from src.cv.letter import letter_prompt_input
from src.cv.match import JobMatch, profile_query_text
from src.cv.pdf_text import extract_cv_text
from src.cv.plan import generate_learning_plan, plan_prompt_input
from src.cv.profile import CVProfile
from src.routes import routes

CV_DIR = Path("tests/eval/cv")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Jombang | 0812-3456-7890 | nama.saya@mail.co.id",
            ["0812-3456-7890", "nama.saya@mail.co.id"],
        ),
        ("Telp (+62) 812 3456 7890", ["(+62) 812 3456 7890"]),
        ("WA +6281234567890 / 0899-1111-2222", ["+6281234567890"]),
        (
            "https://www.linkedin.com/in/nama-saya/ · github.com/NamaSaya/proyek",
            ["linkedin.com/in/nama-saya", "github.com/NamaSaya"],
        ),
        ("Email: a.b+c@mail.co.id.", ["a.b+c@mail.co.id"]),
        ("Kuliah 2019-2023, NIK 3517081234567890, kode 08 2019", []),
        ("", []),
    ],
)
def test_extract_contacts(text, expected):
    assert extract_contacts(text) == expected


def test_contacts_of_eval_cvs_are_phone_then_email():
    for path in sorted(CV_DIR.glob("cv0*.pdf")):
        contacts = extract_contacts(extract_cv_text(path.read_bytes()).text)

        assert len(contacts) == 2, path.name
        assert contacts[0].startswith("+62 812-0000-"), path.name
        assert contacts[1].endswith("@example.com"), path.name


def test_contacts_never_reach_model_inputs():
    contacts = ["0812-3456-7890", "rahasia@mail.com", "linkedin.com/in/rahasia"]
    profile = CVProfile(target_role="Data Analyst", skills=["SQL"], contacts=contacts)
    job = JobMatch(job_id=1, job_title="Data Analyst", score=0.5, semantic=0.5, semantic_scaled=0.5)

    prompts = [
        profile_query_text(profile),
        *plan_prompt_input(profile, compute_skill_gap([job])),
        *letter_prompt_input(profile, job, "Deskripsi."),
        *(message.content for message in planner_messages("Lowongan untukku", "-", profile)),
    ]

    for contact in contacts:
        assert not any(contact in prompt for prompt in prompts)


def test_langfuse_spans_carry_no_cv_data(monkeypatch, make_repository):
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(langfuse, "Langfuse", partial(langfuse.Langfuse, span_exporter=exporter))
    profile = CVProfile(
        full_name="Sekar Wangi",
        target_role="Kurator Museum",
        location="Ternate",
        education_major="Filologi",
        skills=["Origami"],
        contacts=["0812-9999-0000", "sekar@mail.com"],
    )
    job = JobMatch(
        job_id=1,
        job_title="Data Analyst",
        score=0.5,
        semantic=0.5,
        semantic_scaled=0.5,
        missing_skills=["Python"],
    )
    llm = ScriptedLLM(reply=agent_script(plan=plan_reply(("cv_gap", "Skill CV"), schema="CVPlan")))
    repository = make_repository()
    agent = SupervisorAgent(
        llm=llm,
        job_repository=repository,
        rag_agent=RagAgent(llm, repository).create_rag_agent(),
        sql_agent=SqlAgent(llm, repository).create_sql_agent(),
    )
    monkeypatch.setattr(routes, "get_supervisor_agent", lambda: agent)
    cv = {"profile": profile.model_dump(mode="json"), "matches": {"matches": [job.model_dump()]}}
    question = {"role": "user", "content": "Skill apa yang kurang dari CV-ku?"}

    routes.chat_assistant({"history": [], "user_message": question, "cv": cv})
    plan_llm = traced_llm(ScriptedLLM(reply=lambda messages: "-"), "cv_plan")
    generate_learning_plan(profile, compute_skill_gap([job]), llm=plan_llm)
    langfuse.get_client().shutdown()

    spans = exporter.get_finished_spans()
    sent = json.dumps([dict(span.attributes) for span in spans])
    assert {"planner", "cv_agent", "cv_plan"} <= {span.name for span in spans}
    assert "usage_details" in sent
    for private in ["Sekar", "Kurator", "Ternate", "Filologi", "Origami", *profile.contacts]:
        assert private not in sent
