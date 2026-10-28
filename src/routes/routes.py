"""HTTP routes exposed by the Career Assistant API."""

import hmac
import os
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import ValidationError

from src.containers import get_supervisor_agent, get_trace_callbacks
from src.cv.match import MatchResult
from src.cv.profile import CVProfile

router = APIRouter(tags=["chat"])
# Kunci bersama Streamlit dan API; kosong = tanpa pemeriksaan (pengembangan lokal).
API_KEY_ENV = "ASSISTANT_API_KEY"


@router.post("/chat-assistant")
def chat_assistant(
    payload: dict[str, Any], x_api_key: Annotated[str | None, Header()] = None
) -> dict[str, Any]:
    """Jawab pesan pengguna baru beserta riwayat chat dan, jika ada, CV aktif.

    Body: ``history`` (daftar pesan), ``user_message``, dan ``cv`` opsional berisi
    ``{"profile": CVProfile, "matches": MatchResult}`` dalam bentuk JSON. Isi body tidak
    di-log karena memuat profil CV. ``cv`` yang tidak sesuai skema menghasilkan 422.
    Jika env ``ASSISTANT_API_KEY`` terisi, header ``X-API-Key`` wajib sama (401 jika tidak),
    karena URL API publik dan setiap jawaban memakai kredit OpenAI.
    """
    expected = os.getenv(API_KEY_ENV)
    if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="X-API-Key tidak valid.")
    history = payload.get("history", [])
    user_message = payload.get("user_message")
    if not isinstance(history, list) or not isinstance(user_message, dict):
        raise HTTPException(status_code=422, detail="history dan user_message wajib diisi.")
    profile, matches = _cv_context(payload.get("cv"))

    callbacks = get_trace_callbacks()
    state = get_supervisor_agent().graph.invoke(
        {"messages": [*history, user_message], "profile": profile, "matches": matches},
        config={"callbacks": callbacks},
    )
    if callbacks:
        from langfuse import get_client

        # Lambda membekukan proses setelah respons, jadi span yang masih antre dikirim sekarang.
        get_client().flush()
    return {"result": state["result"].model_dump(mode="json")}


def _cv_context(cv: object) -> tuple[CVProfile | None, MatchResult | None]:
    """Ubah bagian ``cv`` body menjadi input graph; ``(None, None)`` jika tidak ada CV."""
    if cv is None:
        return None, None
    try:
        return CVProfile.model_validate(cv["profile"]), MatchResult.model_validate(cv["matches"])
    except (KeyError, TypeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail="cv tidak sesuai skema.") from exc
