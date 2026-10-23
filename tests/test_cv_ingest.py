"""Pintu masuk CV: validasi PDF sebelum LLM, satu panggilan LLM, dan log tanpa data CV.

LLM dipalsukan dengan mengganti ``extract_cv_profile``; PDF memakai CV fiktif set eval.
"""

import logging
from pathlib import Path

from cv import ingest as ingest_module
from cv.ingest import PROFILE_FAILED, ingest_cv
from cv.profile import CVProfile, CVProfileResult
from models.agent import LLMUsage

CV_PDF = Path("tests/eval/cv/cv01_data_analyst.pdf")


def fake_extractor(monkeypatch, result: CVProfileResult) -> list[str]:
    texts: list[str] = []

    def extract(cv_text, *, llm, today=None):
        texts.append(cv_text)
        return result

    monkeypatch.setattr(ingest_module, "extract_cv_profile", extract)
    return texts


def test_rejected_pdf_stops_before_llm(monkeypatch):
    texts = fake_extractor(monkeypatch, CVProfileResult(profile=CVProfile()))
    result = ingest_cv(b"bukan pdf", llm=None)

    assert result.profile is None
    assert result.reason == "not_pdf"
    assert "bukan PDF" in result.error
    assert texts == []  # 0 panggilan LLM


def test_valid_pdf_calls_llm_once_and_returns_profile(monkeypatch):
    usage = LLMUsage(llm_calls=1, input_tokens=900, output_tokens=200, total_tokens=1100)
    profile = CVProfile(target_role="Data Analyst", skills=["SQL"])
    texts = fake_extractor(monkeypatch, CVProfileResult(profile=profile, usage=usage))

    result = ingest_cv(CV_PDF.read_bytes(), llm=None)

    assert result.error is None
    assert result.profile == profile
    assert result.usage.total_tokens == 1100
    assert len(texts) == 1 and "Data Analyst" in texts[0]
    assert (result.pages, result.chars, result.truncated) == (1, len(texts[0]), False)
    assert "Lintas Niaga" in texts[0]
    assert "Lintas Niaga" not in result.model_dump_json()  # teks CV tidak ikut dikembalikan


def test_profile_failure_is_friendly_and_logs_no_cv_data(monkeypatch, caplog):
    secret = "Output LLM tidak valid: {'full_name': 'BIMA ARDITYA KUSUMA'}"
    fake_extractor(monkeypatch, CVProfileResult(usage=LLMUsage(llm_calls=1), error=secret))

    with caplog.at_level(logging.WARNING, logger="cv.ingest"):
        result = ingest_cv(CV_PDF.read_bytes(), llm=None)

    assert result.profile is None
    assert result.reason == "profile"
    assert result.error == PROFILE_FAILED
    assert result.usage.llm_calls == 1
    assert caplog.records  # kegagalan tetap tercatat
    assert "BIMA" not in caplog.text and "Data Analyst" not in caplog.text
