import os
import asyncio
import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from db.database import get_db
from db import queries
from agent import tools as tool_fns

router = APIRouter()

TAVUS_BASE = "https://tavusapi.com/v2"


def _headers() -> dict:
    key = os.getenv("TAVUS_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="TAVUS_API_KEY not configured")
    return {"x-api-key": key, "Content-Type": "application/json"}


async def _get_replica_id() -> str:
    """Use TAVUS_REPLICA_ID if set, otherwise pick the first available replica."""
    replica_id = os.getenv("TAVUS_REPLICA_ID")
    if replica_id:
        return replica_id

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{TAVUS_BASE}/replicas", headers=_headers())
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Tavus replicas fetch failed: {r.text}")
        data = r.json()
        replicas = data.get("data", data) if isinstance(data, dict) else data
        if not replicas:
            raise HTTPException(status_code=502, detail="No Tavus replicas available on this account")
        return replicas[0]["replica_id"]


class ConversationResponse(BaseModel):
    conversation_id: str
    conversation_url: str


@router.post("/tavus/conversation", response_model=ConversationResponse)
async def create_tavus_conversation(session_id: str):
    """Create a Tavus CVI conversation and return the embeddable URL."""
    db = await get_db()
    await queries.create_session(db, session_id)

    replica_id = await _get_replica_id()
    persona_id = os.getenv("TAVUS_PERSONA_ID") or await _create_persona(replica_id)

    payload = {
        "replica_id": replica_id,
        "persona_id": persona_id,
        "conversation_name": f"mykare-{session_id}",
        "conversational_context": (
            "You are Maya, a friendly healthcare front-desk assistant for Mykare. "
            "You help patients book, cancel, and modify doctor appointments. "
            "Be warm, clear, and concise. Always identify the patient by phone "
            "before appointment actions. Use tools for slot lookup, booking, "
            "retrieval, cancellation, modification, and ending the conversation."
        ),
        "custom_greeting": (
            "Hello! I'm Maya, your Mykare healthcare assistant. "
            "How can I help you today?"
        ),
        "properties": {
            "max_call_duration": 3600,
            "enable_recording": False,
        },
    }
    public_base_url = os.getenv("PUBLIC_BASE_URL")
    if public_base_url:
        payload["callback_url"] = f"{public_base_url.rstrip('/')}/api/tavus/webhook"

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{TAVUS_BASE}/conversations",
            headers=_headers(),
            json=payload,
        )

    if r.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"Tavus conversation creation failed: {r.text}")

    body = r.json()
    await queries.link_session_tavus_conversation(db, session_id, body["conversation_id"])
    return ConversationResponse(
        conversation_id=body["conversation_id"],
        conversation_url=body["conversation_url"],
    )


@router.delete("/tavus/conversation/{conversation_id}")
async def end_tavus_conversation(conversation_id: str, session_id: str | None = None):
    """End a Tavus conversation when the call ends."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            f"{TAVUS_BASE}/conversations/{conversation_id}/end",
            headers=_headers(),
        )
    if r.status_code not in (200, 204):
        # Older Tavus accounts may still accept DELETE.
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.delete(
                f"{TAVUS_BASE}/conversations/{conversation_id}",
                headers=_headers(),
            )
        if r.status_code not in (200, 204):
            return {"status": "warning", "detail": r.text}

    await _sync_tavus_transcript(conversation_id, session_id=session_id)
    return {"status": "ended"}


@router.post("/tavus/webhook")
async def tavus_webhook(request: Request):
    payload = await request.json()
    if payload.get("event_type") == "application.transcription_ready":
        conversation_id = payload.get("conversation_id")
        transcript = (payload.get("properties") or {}).get("transcript") or []
        if conversation_id and transcript:
            await _save_transcript_summary(conversation_id, transcript)
    return {"status": "ok"}


async def _create_persona(replica_id: str) -> str:
    payload = {
        "persona_name": "Mykare Front Desk",
        "pipeline_mode": "full",
        "default_replica_id": replica_id,
        "system_prompt": (
            "You are Maya, a healthcare front-desk voice agent. Ask for name "
            "and phone number, maintain context, and use the provided tools for "
            "all appointment actions. Confirm dates and times clearly."
        ),
        "layers": {
            "llm": {
                "model": os.getenv("TAVUS_LLM_MODEL", "tavus-gpt-oss"),
                "tools": _tavus_tool_schemas(),
            }
        },
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{TAVUS_BASE}/personas", headers=_headers(), json=payload)
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"Tavus persona creation failed: {r.text}")
    return r.json()["persona_id"]


async def _sync_tavus_transcript(conversation_id: str, session_id: str | None = None) -> None:
    for _ in range(5):
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{TAVUS_BASE}/conversations/{conversation_id}?verbose=true",
                headers=_headers(),
            )
        if r.status_code == 200:
            body = r.json()
            transcript = body.get("transcript") or (body.get("properties") or {}).get("transcript")
            if transcript:
                await _save_transcript_summary(conversation_id, transcript, session_id=session_id)
                return
        await asyncio.sleep(2)

    if session_id:
        db = await get_db()
        session = await queries.get_session(db, session_id)
        if session and not session.summary:
            await tool_fns.end_conversation(session_id, session.user_id, [])


async def _save_transcript_summary(
    conversation_id: str,
    transcript: list[dict],
    session_id: str | None = None,
) -> None:
    db = await get_db()
    session = await queries.get_session(db, session_id) if session_id else None
    if not session:
        session = await queries.get_session_by_tavus_conversation(db, conversation_id)
    if not session:
        return
    cleaned = [
        {
            "role": "assistant" if item.get("role") in ("assistant", "replica") else "user",
            "content": item.get("content") or item.get("speech") or "",
            "ts": item.get("timestamp") or item.get("ts"),
        }
        for item in transcript
        if item.get("role") in ("user", "assistant", "replica")
    ]
    await tool_fns.end_conversation(session.session_id, session.user_id, cleaned)


def _tavus_tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "identify_user",
                "description": "Look up or create a patient by phone number. Call this before appointment actions.",
                "parameters": {
                    "type": "object",
                    "properties": {"phone": {"type": "string"}},
                    "required": ["phone"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_user_name",
                "description": "Save the patient's name after they provide it.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_slots",
                "description": "Return available appointment slots for a date.",
                "parameters": {
                    "type": "object",
                    "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                    "required": ["date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "book_appointment",
                "description": "Book the confirmed date and time for the identified patient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "time_slot": {"type": "string", "description": "HH:MM"},
                    },
                    "required": ["date", "time_slot"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "retrieve_appointments",
                "description": "Show the identified patient's current appointments.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_appointment",
                "description": "Cancel an appointment by ID after confirming with the patient.",
                "parameters": {
                    "type": "object",
                    "properties": {"appointment_id": {"type": "integer"}},
                    "required": ["appointment_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "modify_appointment",
                "description": "Move an appointment to a new date and time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_id": {"type": "integer"},
                        "new_date": {"type": "string"},
                        "new_time": {"type": "string"},
                    },
                    "required": ["appointment_id", "new_date", "new_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "end_conversation",
                "description": "End the conversation and trigger final summary generation.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]
