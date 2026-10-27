"""Kontak dari teks CV (regex, tanpa LLM) dan jaminan bahwa kontak tidak dikirim ke model."""

from pathlib import Path

import pytest

from src.agents.supervisor_agent_graph import planner_messages
from src.cv.contacts import extract_contacts
from src.cv.gap import compute_skill_gap
from src.cv.letter import letter_prompt_input
from src.cv.match import JobMatch, profile_query_text
from src.cv.pdf_text import extract_cv_text
from src.cv.plan import plan_prompt_input
from src.cv.profile import CVProfile

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
