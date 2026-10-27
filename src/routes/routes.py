"""HTTP routes exposed by the Career Assistant API."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from src.containers import get_supervisor_agent
from src.cv.match import MatchResult
from src.cv.profile import CVProfile

router = APIRouter(tags=["chat"])


@router.post("/chat-assistant")
def chat_assistant(payload: dict[str, Any]) -> dict[str, Any]:
    """Jawab pesan pengguna baru beserta riwayat chat dan, jika ada, CV aktif.

    Body: ``history`` (daftar pesan), ``user_message``, dan ``cv`` opsional berisi
    ``{"profile": CVProfile, "matches": MatchResult}`` dalam bentuk JSON. Isi body tidak
    di-log karena memuat profil CV. ``cv`` yang tidak sesuai skema menghasilkan 422.
    """
    history = payload.get("history", [])
    user_message = payload.get("user_message")
    if not isinstance(history, list) or not isinstance(user_message, dict):
        raise HTTPException(status_code=422, detail="history dan user_message wajib diisi.")
    profile, matches = _cv_context(payload.get("cv"))

    state = get_supervisor_agent().graph.invoke(
        {"messages": [*history, user_message], "profile": profile, "matches": matches}
    )
    return {"result": state["result"].model_dump(mode="json")}


def _cv_context(cv: object) -> tuple[CVProfile | None, MatchResult | None]:
    """Ubah bagian ``cv`` body menjadi input graph; ``(None, None)`` jika tidak ada CV."""
    if cv is None:
        return None, None
    try:
        return CVProfile.model_validate(cv["profile"]), MatchResult.model_validate(cv["matches"])
    except (KeyError, TypeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail="cv tidak sesuai skema.") from exc
