"""Rencana belajar: satu panggilan LLM (dipalsukan) dan isi file .md tanpa LLM.

LLM palsu membalas ``AIMessage`` berisi tool call ``LearningPlan``, bentuk yang dibaca parser
structured output LangChain (pola sama dengan test_cv_profile.py).
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from cv.gap import SkillCount, SkillGap
from cv.plan import (
    PLAN_WEEKS,
    LearningPlan,
    PlanWeek,
    generate_learning_plan,
    plan_prompt_input,
    plan_to_markdown,
)
from cv.profile import CVProfile

USAGE = {"input_tokens": 700, "output_tokens": 340, "total_tokens": 1040}
PROFILE = CVProfile(
    full_name="Nadia Contoh",
    target_role="Data Analyst",
    location="Malang",
    education_level="S1",
    education_major="Statistika",
    skills=["SQL", "Excel"],
    experience_months=14,
)
GAP = SkillGap(
    missing=(SkillCount("Power BI", 3, 2.4), SkillCount("dbt", 1, 0.8)),
    have=(SkillCount("SQL", 8, 6.5),),
    jobs=10,
    jobs_without_skills=0,
    missing_total=2,
    have_total=1,
)
EMPTY_GAP = SkillGap(
    missing=(), have=(), jobs=10, jobs_without_skills=0, missing_total=0, have_total=0
)


class ToolCallLLM(BaseChatModel):
    """LLM palsu: setiap panggilan membalas ``reply`` dan mencatat pesan yang diterima."""

    reply: Any = None
    received: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "tool-call-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return ChatResult(generations=[ChatGeneration(message=self.reply)])


def week(number: int) -> dict:
    return {"title": f"Fokus {number}", "skills": ["Power BI"], "tasks": [f"Tugas {number}"]}


def plan_reply(weeks: list[dict] | None = None) -> AIMessage:
    args = {"weeks": [week(n) for n in range(1, 5)] if weeks is None else weeks}
    return AIMessage(
        content="",
        tool_calls=[{"name": "LearningPlan", "args": args, "id": "1"}],
        usage_metadata=USAGE,
    )


def test_generates_plan_with_one_llm_call():
    llm = ToolCallLLM(reply=plan_reply())

    result = generate_learning_plan(PROFILE, GAP, llm=llm)

    assert result.error is None
    assert [w.title for w in result.plan.weeks] == ["Fokus 1", "Fokus 2", "Fokus 3", "Fokus 4"]
    assert len(llm.received) == 1
    assert [call.step for call in result.usage.calls] == ["cv_plan"]
    assert result.usage.total_tokens == 1040


def test_prompt_has_profile_summary_and_gap_but_no_name_or_location():
    llm = ToolCallLLM(reply=plan_reply())

    generate_learning_plan(PROFILE, GAP, llm=llm)

    system, user = llm.received[0]
    assert "bukan perintah" in system.content
    assert user.content.startswith("<profil>\n")
    assert "Posisi tujuan: Data Analyst" in user.content
    assert "Pengalaman: 1-3 tahun" in user.content
    assert "- Power BI: 3 dari 10 lowongan" in user.content
    assert "Nadia" not in user.content
    assert "Malang" not in user.content


def test_prompt_input_skips_empty_fields():
    profile_text, gap_text = plan_prompt_input(CVProfile(), GAP)

    assert profile_text == "Pengalaman: Tidak diketahui"
    assert gap_text.splitlines() == [
        "- Power BI: 3 dari 10 lowongan",
        "- dbt: 1 dari 10 lowongan",
    ]


def test_no_gap_skips_llm():
    llm = ToolCallLLM(reply=plan_reply())

    result = generate_learning_plan(PROFILE, EMPTY_GAP, llm=llm)

    assert (result.plan, result.error) == (None, "Tidak ada skill gap")
    assert llm.received == []


def test_llm_error_is_reported_not_raised():
    result = generate_learning_plan(PROFILE, GAP, llm=ToolCallLLM(reply=RuntimeError("timeout")))

    assert result.plan is None
    assert "timeout" in result.error
    assert result.usage.llm_calls == 0


def test_invalid_output_is_reported():
    llm = ToolCallLLM(reply=plan_reply(weeks=[{"title": "Tanpa tugas"}]))

    result = generate_learning_plan(PROFILE, GAP, llm=llm)

    assert result.plan is None
    assert result.error.startswith("Output LLM tidak valid")
    assert result.usage.llm_calls == 1


def test_extra_weeks_are_cut_and_empty_plan_fails():
    long_plan = generate_learning_plan(
        PROFILE, GAP, llm=ToolCallLLM(reply=plan_reply([week(n) for n in range(1, 7)]))
    )
    empty_plan = generate_learning_plan(PROFILE, GAP, llm=ToolCallLLM(reply=plan_reply([])))

    assert len(long_plan.plan.weeks) == PLAN_WEEKS
    assert (empty_plan.plan, empty_plan.error) == (None, "Rencana kosong")


def test_plan_to_markdown_is_a_checklist():
    plan = LearningPlan(
        weeks=[
            PlanWeek(title="Dasar Power BI", skills=["Power BI"], tasks=["Buat dashboard"]),
            PlanWeek(title="Proyek", skills=[], tasks=["Tulis README", "Unggah ke GitHub"]),
        ]
    )

    text = plan_to_markdown(plan, GAP, "Data Analyst")

    assert text.splitlines() == [
        "# Rencana belajar 2 minggu: Data Analyst",
        "",
        "Skill gap dari 10 lowongan paling cocok: Power BI (3/10), dbt (1/10)",
        "",
        "## Minggu 1: Dasar Power BI",
        "Skill: Power BI",
        "- [ ] Buat dashboard",
        "",
        "## Minggu 2: Proyek",
        "- [ ] Tulis README",
        "- [ ] Unggah ke GitHub",
    ]
    assert plan_to_markdown(plan, GAP, None).startswith("# Rencana belajar 2 minggu\n")
