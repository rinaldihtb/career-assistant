"""HTTP routes exposed by the Career Assistant API."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.containers import get_supervisor_agent

router = APIRouter(tags=["chat"])


@router.post("/chat-assistant")
def chat_assistant(payload: dict[str, Any]) -> dict[str, Any]:
    """Answer a new user message with its prior chat history."""
    history = payload.get("history", [])
    user_message = payload.get("user_message")
    if not isinstance(history, list) or not isinstance(user_message, dict):
        raise HTTPException(status_code=422, detail="history dan user_message wajib diisi.")

    state = get_supervisor_agent().graph.invoke({"messages": [*history, user_message]})
    return {"result": state["result"].model_dump(mode="json")}
