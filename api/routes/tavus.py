import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
    replica_id = await _get_replica_id()

    payload = {
        "replica_id": replica_id,
        "conversation_name": f"mykare-{session_id}",
        "conversational_context": (
            "You are Maya, a friendly healthcare front-desk assistant for Mykare. "
            "You help patients book, cancel, and modify doctor appointments. "
            "Be warm, clear, and concise."
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

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{TAVUS_BASE}/conversations",
            headers=_headers(),
            json=payload,
        )

    if r.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"Tavus conversation creation failed: {r.text}")

    body = r.json()
    return ConversationResponse(
        conversation_id=body["conversation_id"],
        conversation_url=body["conversation_url"],
    )


@router.delete("/tavus/conversation/{conversation_id}")
async def end_tavus_conversation(conversation_id: str):
    """End a Tavus conversation when the call ends."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.delete(
            f"{TAVUS_BASE}/conversations/{conversation_id}",
            headers=_headers(),
        )
    if r.status_code not in (200, 204):
        # Non-fatal — log but don't fail
        return {"status": "warning", "detail": r.text}
    return {"status": "ended"}
