"""LLMUsage: pencatatan per panggilan (langkah, model, token) dan penggabungan."""

from langchain_core.messages import AIMessage

from src.models.agent import LLMUsage


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
