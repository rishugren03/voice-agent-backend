"""
Tool implementations called by GPT-4o during a conversation.
Each function returns a plain dict — GPT-4o receives this as the tool result.
All DB access goes through db/queries.py — no SQL here.
"""
import json
import datetime
import os
import re

from openai import AsyncOpenAI

from db.database import get_db
from db import queries


async def identify_user(phone: str) -> dict:
    """Upsert user by phone number. Returns user_id used for all subsequent calls."""
    db = await get_db()
    user = await queries.upsert_user(db, phone)
    return {"user_id": user.id, "name": user.name, "phone": user.phone, "is_new": user.name is None}


async def set_user_name(user_id: int, name: str) -> dict:
    """Save the user's name once extracted from conversation."""
    db = await get_db()
    await queries.update_user_name(db, user_id, name)
    return {"success": True}


async def fetch_slots(date: str) -> dict:
    """Return available appointment slots for a given date (YYYY-MM-DD)."""
    db = await get_db()
    available = await queries.get_available_slots(db, date)
    return {"date": date, "available_slots": available, "count": len(available)}


async def book_appointment(user_id: int, date: str, time_slot: str) -> dict:
    """Book an appointment. The DB unique index prevents double-booking."""
    db = await get_db()
    appt, err = await queries.create_appointment(db, user_id, date, time_slot)
    if err:
        return {"success": False, "reason": err}
    return {
        "success": True,
        "appointment_id": appt.id,
        "date": appt.date,
        "time_slot": appt.time_slot,
        "doctor": appt.doctor,
    }


async def retrieve_appointments(user_id: int) -> dict:
    """Fetch all confirmed appointments for a user — single JOIN query."""
    db = await get_db()
    appointments = await queries.get_user_appointments(db, user_id)
    return {
        "appointments": [a.to_dict() for a in appointments],
        "count": len(appointments),
    }


async def cancel_appointment(user_id: int, appointment_id: int) -> dict:
    """Soft-cancel an appointment (sets status to 'cancelled')."""
    db = await get_db()
    ok = await queries.cancel_appointment(db, user_id, appointment_id)
    if not ok:
        return {"success": False, "reason": "Appointment not found or already cancelled."}
    return {"success": True}


async def modify_appointment(
    user_id: int, appointment_id: int, new_date: str, new_time: str
) -> dict:
    """Atomically cancel old appointment and create a new one."""
    db = await get_db()
    appt, err = await queries.modify_appointment(db, user_id, appointment_id, new_date, new_time)
    if err:
        return {"success": False, "reason": err}
    return {
        "success": True,
        "new_appointment_id": appt.id,
        "date": appt.date,
        "time_slot": appt.time_slot,
        "doctor": appt.doctor,
    }


async def generate_summary(transcript: list[dict]) -> dict:
    """
    Generate a structured call summary via GPT-4o.
    Returns the summary dict — DB saving is the caller's responsibility.
    """
    # If transcript is empty, return a minimal summary without calling GPT
    if not transcript:
        return {
            "overview": "No conversation recorded.",
            "appointments": [],
            "extracted": {"name": None, "phone": None, "intent": None},
            "preferences": None,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }

    if not os.getenv("OPENAI_API_KEY"):
        text = " ".join(str(t.get("content", "")) for t in transcript)
        return {
            "overview": text[:220] or "Conversation completed.",
            "appointments": [],
            "extracted": {
                "name": None,
                "phone": _extract_phone(text),
                "intent": _extract_intent(text),
            },
            "preferences": None,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }

    client = AsyncOpenAI()
    summary_prompt = [
        {
            "role": "system",
            "content": (
                "You are a medical receptionist summarising a voice call. "
                "Return ONLY valid JSON matching exactly this schema:\n"
                '{"overview": "string", '
                '"appointments": [{"action": "booked|cancelled|modified", "date": "string", "time": "string", "doctor": "string"}], '
                '"extracted": {"name": "string or null", "phone": "string or null", "intent": "string"}, '
                '"preferences": "string or null"}'
            ),
        },
        {
            "role": "user",
            "content": f"Summarise this call transcript:\n{json.dumps(transcript, indent=2)}",
        },
    ]

    response = await client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=summary_prompt,
        response_format={"type": "json_object"},
        max_tokens=512,
    )

    summary = json.loads(response.choices[0].message.content)
    summary["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return summary


async def end_conversation(
    session_id: str,
    user_id: int | None,
    transcript: list[dict],
    cost_breakdown: dict | None = None,
) -> dict:
    """Generate and persist the final summary for a call session."""
    db = await get_db()
    summary = await generate_summary(transcript)
    await queries.save_session_summary(
        db,
        session_id=session_id,
        transcript=json.dumps(transcript),
        summary=json.dumps(summary),
        cost_breakdown=json.dumps(cost_breakdown or _zero_cost()),
    )
    if user_id:
        await queries.link_session_user(db, session_id, user_id)
    return {"summary": summary}


def _zero_cost() -> dict:
    return {"stt_usd": 0, "tts_usd": 0, "llm_usd": 0, "total_usd": 0}


def _extract_phone(text: str) -> str | None:
    match = re.search(r"(?<!\d)(?:\+?91[-\s]?)?([6-9]\d{9})(?!\d)", text)
    return match.group(1) if match else None


def _extract_intent(text: str) -> str | None:
    lowered = text.lower()
    if "cancel" in lowered:
        return "cancel_appointment"
    if "reschedule" in lowered or "modify" in lowered or "change" in lowered:
        return "modify_appointment"
    if "book" in lowered or "schedule" in lowered:
        return "book_appointment"
    if "appointment" in lowered and ("show" in lowered or "view" in lowered or "have" in lowered):
        return "retrieve_appointments"
    return None


# ── GPT-4o function schemas ───────────────────────────────────────────────────
# These are passed to the OpenAI API as the `tools` parameter.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "identify_user",
            "description": "Look up or create a user by phone number. Always call this first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "description": "User's phone number"},
                },
                "required": ["phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_user_name",
            "description": "Save the user's name once you learn it from conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "name": {"type": "string"},
                },
                "required": ["user_id", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_slots",
            "description": "Get available appointment slots for a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time_slot": {"type": "string", "description": "HH:MM (24h), e.g. 09:30"},
                },
                "required": ["user_id", "date", "time_slot"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_appointments",
            "description": "Get all confirmed appointments for a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel a specific appointment by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "appointment_id": {"type": "integer"},
                },
                "required": ["user_id", "appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_appointment",
            "description": "Change an existing appointment to a new date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "appointment_id": {"type": "integer"},
                    "new_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "new_time": {"type": "string", "description": "HH:MM (24h)"},
                },
                "required": ["user_id", "appointment_id", "new_date", "new_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": "End the call, generate a summary, and save the session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "user_id": {"type": "integer", "description": "null if user was never identified"},
                    "transcript": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["session_id", "transcript"],
            },
        },
    },
]


# ── Dispatcher ─────────────────────────────────────────────────────────────────
# Maps tool name → function so the agent loop can call them by name.

TOOL_REGISTRY = {
    "identify_user": identify_user,
    "set_user_name": set_user_name,
    "fetch_slots": fetch_slots,
    "book_appointment": book_appointment,
    "retrieve_appointments": retrieve_appointments,
    "cancel_appointment": cancel_appointment,
    "modify_appointment": modify_appointment,
    "generate_summary": generate_summary,
    "end_conversation": end_conversation,
}
