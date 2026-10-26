"""Test pencocokan skill CV vs deskripsi lowongan (tanpa LLM, tanpa database)."""

import re

import pytest

from src.cv import skills
from src.cv.skills import (
    SKILL_VOCAB,
    build_cv_skills,
    match_skills,
    skill_variants,
    term_pattern,
    vocab_in_text,
)


def test_vocab_aliases_unique_across_entries():
    seen: dict[str, str] = {}
    for name, aliases in SKILL_VOCAB.items():
        for alias in aliases:
            assert alias == alias.lower(), alias
            assert alias not in seen, f"{alias} ada di {seen.get(alias)} dan {name}"
            seen[alias] = name


@pytest.mark.parametrize(
    ("alias", "text", "expected"),
    [
        ("sql", "Menguasai SQL dan Python", True),
        ("sql", "Pengalaman MySQL atau NoSQL", False),
        ("c++", "Skilled in C++ and Go", True),
        ("c#", "C#, .NET", True),
        ("power bi", "Dashboard Power-BI", True),
        ("power bi", "power  bi", True),
        ("java", "Python, Java, or Scala", True),
        ("java", "JavaScript developer", False),
        ("java", "Based in Bekasi, West Java", False),
        ("java", "PT Cahaya Murni Central Java", False),
        ("recruitment", "Handle end-to-end recruitment for all roles", True),
        ("recruitment", "We never charge fees during the recruitment process", False),
        ("recruitment", "Anypear is a recruitment agency", False),
    ],
)
def test_term_pattern_word_boundaries_and_measured_exclusions(alias, text, expected):
    assert bool(term_pattern(alias).search(text)) is expected


def test_vocab_in_text_finds_display_names():
    found = vocab_in_text("Proficient in Microsoft Excel, Power BI, and SQL. Fluent in English.")
    assert found == {"Excel", "Power BI", "SQL", "English"}
    assert vocab_in_text(None) == frozenset()
    assert vocab_in_text("") == frozenset()


def test_vocab_prefilter_gives_same_result_as_full_regex():
    text = "UI/UX, Generative AI, RAG pipelines, labour law, social media management, PowerBI"
    full = {name for name, _, pattern in skills._VOCAB_INDEX if pattern.search(text)}
    assert vocab_in_text(text) == full


@pytest.mark.parametrize(
    ("skill", "expected"),
    [
        ("Computer Vision (YOLO)", ["Computer Vision", "YOLO"]),
        ("English (Intermediate)", ["English"]),
        ("HTML/CSS", ["HTML", "CSS"]),
        ("A/B Testing", ["A/B Testing"]),
        ("CI/CD (GitHub Actions)", ["CI/CD", "GitHub Actions"]),
        ("UI/UX Design", ["UI/UX Design"]),
        ("Bahasa Indonesia", []),
        ("Python (Pandas, NumPy)", ["Python", "Pandas", "NumPy"]),
        ("Communication", []),
        ("R", []),
        ("  Machine   Learning ", ["Machine Learning"]),
    ],
)
def test_skill_variants(skill, expected):
    assert skill_variants(skill) == expected


def test_build_cv_skills_maps_vocab_and_keeps_extras():
    cv = build_cv_skills(
        ["Python (Pandas, NumPy)", "Computer Vision (YOLO)", "SHAP (Explainable AI)", "Teamwork"]
    )
    assert cv.vocab == {"Python", "Pandas", "NumPy", "Computer Vision"}
    assert [display for display, _ in cv.extras] == ["YOLO", "SHAP", "Explainable AI"]


def test_build_cv_skills_cv_only_alias_word():
    cv = build_cv_skills(["Microsoft Office (Word, Excel)", "Go"])
    assert cv.vocab == {"Microsoft Office", "Microsoft Word", "Excel", "Golang"}
    assert cv.extras == ()


def test_build_cv_skills_keeps_slash_terms_and_drops_fragments():
    cv = build_cv_skills(["A/B Testing", "SAP FI/CO", "Inggris", "Indonesia"])
    assert cv.vocab == {"A/B Testing", "SAP", "English"}
    assert cv.extras == ()


@pytest.mark.parametrize(
    "text", ["Mampu berbahasa Inggris dengan baik", "Bahasa Inggris aktif", "Fluent in English"]
)
def test_english_vocab_aliases(text):
    assert "English" in vocab_in_text(text)


def test_match_skills_ab_testing_is_matched_not_missing():
    cv = build_cv_skills(["A/B Testing", "SQL"])
    description = "Melakukan A/B testing kampanye dan analisis dengan SQL."

    result = match_skills(cv, vocab_in_text(description), description)

    assert set(result.matched) == {"A/B Testing", "SQL"}
    assert result.missing == ()


def test_match_skills_portion_and_missing_sorted_by_frequency():
    description = "Requirements: SQL, Python, Tableau, Power BI, and experience with YOLO models."
    cv = build_cv_skills(["Python", "SQL", "Computer Vision (YOLO)"])
    result = match_skills(
        cv,
        vocab_in_text(description),
        description,
        frequency={"SQL": 46, "Python": 43, "Power BI": 30, "Tableau": 24},
    )
    assert result.matched == ("SQL", "Python", "YOLO")
    assert result.missing == ("Power BI", "Tableau")
    assert result.required == 5
    assert result.portion == pytest.approx(3 / 5)


def test_match_skills_without_frequency_sorts_alphabetically():
    description = "Tableau, Power BI, Excel"
    result = match_skills(build_cv_skills([]), vocab_in_text(description), description)
    assert result.missing == ("Excel", "Power BI", "Tableau")
    assert result.portion == 0.0


def test_match_skills_portion_none_when_job_mentions_no_skill():
    result = match_skills(build_cv_skills(["Python"]), frozenset(), "Ramah dan teliti.")
    assert result.required == 0
    assert result.portion is None


def test_extra_skill_pattern_escapes_regex_characters():
    cv = build_cv_skills(["Node-RED (v3.x)"])
    assert all(isinstance(pattern, re.Pattern) for _, pattern in cv.extras)
    result = match_skills(cv, frozenset(), "Familiar with Node RED flows")
    assert result.matched == ("Node-RED",)
