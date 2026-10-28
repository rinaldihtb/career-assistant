"""LLMUsage: pencatatan per panggilan (langkah, model, token), penggabungan, dan biaya."""

import pytest
from langchain_core.messages import AIMessage

from src.helpers.format import format_usage
from src.models.agent import LLMCall, LLMUsage


def test_add_records_step_model_and_tokens():
    usage = LLMUsage()
    usage.add(
        AIMessage(
            "ok",
            usage_metadata={"input_tokens": 120, "output_tokens": 20, "total_tokens": 140},
            response_metadata={"model_name": "gpt-4o-mini-2024-07-18"},
        ),
        step="router",
    )

    assert usage.llm_calls == 1
    assert usage.total_tokens == 140
    assert usage.calls[0].model_dump() == {
        "step": "router",
        "model": "gpt-4o-mini-2024-07-18",
        "input_tokens": 120,
        "output_tokens": 20,
        "total_tokens": 140,
    }


def test_add_without_metadata_uses_defaults():
    usage = LLMUsage()
    usage.add(AIMessage("ok"))

    assert usage.calls[0].step == "llm"
    assert usage.calls[0].model == "-"
    assert usage.total_tokens == 0


def test_merged_keeps_call_order():
    first, second = LLMUsage(), LLMUsage()
    first.add(AIMessage("a"), step="router")
    second.add(AIMessage("b"), step="sql_generate")
    second.add(AIMessage("c"), step="sql_answer")

    merged = first.merged(second)

    assert [call.step for call in merged.calls] == ["router", "sql_generate", "sql_answer"]
    assert merged.llm_calls == 3
    assert len(first.calls) == 1  # input tidak diubah


def _call(model: str, input_tokens: int, output_tokens: int) -> LLMCall:
    return LLMCall(
        step="planner",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def test_cost_uses_price_of_model_snapshot():
    call = _call("gpt-4o-mini-2024-07-18", 1000, 200)

    assert call.cost_usd == pytest.approx(0.00027)  # 1000 x 0.15 + 200 x 0.60 per juta token
    assert _call("model-lain", 1000, 200).cost_usd is None


def test_usage_cost_is_unknown_when_a_price_or_call_detail_is_missing():
    known = _call("gpt-4o-mini-2024-07-18", 1000, 200)

    assert LLMUsage(llm_calls=2, calls=[known, known]).cost_usd == pytest.approx(0.00054)
    assert LLMUsage(llm_calls=2, calls=[known, _call("-", 10, 5)]).cost_usd is None
    assert LLMUsage(llm_calls=1).cost_usd is None
    assert LLMUsage().cost_usd == 0


def test_format_usage_shows_calls_model_tokens_and_cost():
    usage = LLMUsage()
    usage.add(
        AIMessage(
            "ok",
            usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
            response_metadata={"model_name": "gpt-4o-mini-2024-07-18"},
        ),
        step="cv_profile",
    )

    assert format_usage(usage) == (
        "1 panggilan LLM (gpt-4o-mini-2024-07-18) · 1.000 token masuk, 200 keluar · $0.000270"
    )
    assert format_usage(LLMUsage()) == "0 panggilan LLM · 0 token masuk, 0 keluar · $0.000000"
