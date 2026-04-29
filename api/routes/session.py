import json
import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from db.database import get_db
from db import queries
from agent import tools as tool_fns

router = APIRouter()


class ToolCallRequest(BaseModel):
    tool: str
    arguments: dict = Field(default_factory=dict)


class FinishSessionRequest(BaseModel):
    transcript: list[dict] = Field(default_factory=list)


def _json_or_none(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


@router.get("/session/{session_id}/summary")
async def get_summary(session_id: str):
    db = await get_db()
    session = await queries.get_session(db, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.summary:
        raise HTTPException(status_code=202, detail="Summary not ready yet")

    return {
        "session_id": session_id,
        "summary": json.loads(session.summary),
        "cost_breakdown": json.loads(session.cost_breakdown) if session.cost_breakdown else None,
        "transcript": _json_or_none(session.transcript),
        "started_at": session.started_at,
        "ended_at": session.ended_at,
    }


@router.post("/session/{session_id}/tool-call")
async def execute_tool_call(session_id: str, req: ToolCallRequest):
    """Execute one appointment tool for Tavus/browser/agent event handlers."""
    db = await get_db()
    session = await queries.get_session(db, session_id)
    if not session:
        await queries.create_session(db, session_id)
        session = await queries.get_session(db, session_id)

    tool = req.tool
    args = dict(req.arguments or {})

    if tool == "identify_user":
        phone = str(args.get("phone") or "")
        if not phone:
            raise HTTPException(status_code=400, detail="identify_user requires phone")
        result = await tool_fns.identify_user(phone)
        await queries.link_session_user(db, session_id, result["user_id"])
    elif tool == "set_user_name":
        if not session or not session.user_id:
            raise HTTPException(status_code=400, detail="identify_user must run before set_user_name")
        result = await tool_fns.set_user_name(session.user_id, str(args.get("name") or ""))
    elif tool == "fetch_slots":
        result = await tool_fns.fetch_slots(str(args.get("date") or ""))
    elif tool == "book_appointment":
        if not session or not session.user_id:
            raise HTTPException(status_code=400, detail="identify_user must run before book_appointment")
        result = await tool_fns.book_appointment(
            session.user_id,
            str(args.get("date") or ""),
            str(args.get("time_slot") or args.get("time") or ""),
        )
    elif tool == "retrieve_appointments":
        if not session or not session.user_id:
            raise HTTPException(status_code=400, detail="identify_user must run before retrieve_appointments")
        result = await tool_fns.retrieve_appointments(session.user_id)
    elif tool == "cancel_appointment":
        if not session or not session.user_id:
            raise HTTPException(status_code=400, detail="identify_user must run before cancel_appointment")
        result = await tool_fns.cancel_appointment(session.user_id, int(args.get("appointment_id") or 0))
    elif tool == "modify_appointment":
        if not session or not session.user_id:
            raise HTTPException(status_code=400, detail="identify_user must run before modify_appointment")
        result = await tool_fns.modify_appointment(
            session.user_id,
            int(args.get("appointment_id") or 0),
            str(args.get("new_date") or args.get("date") or ""),
            str(args.get("new_time") or args.get("time_slot") or args.get("time") or ""),
        )
    elif tool == "end_conversation":
        transcript = args.get("transcript") if isinstance(args.get("transcript"), list) else []
        user_id = session.user_id if session else None
        result = await tool_fns.end_conversation(session_id, user_id, transcript)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool}")

    return {
        "tool": tool,
        "status": "done" if result.get("success", True) else "error",
        "display": _tool_display(tool, result),
        "result": result,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


@router.post("/session/{session_id}/finish")
async def finish_session(session_id: str, req: FinishSessionRequest):
    db = await get_db()
    session = await queries.get_session(db, session_id)
    if not session:
        await queries.create_session(db, session_id)
        session = await queries.get_session(db, session_id)
    if session and session.summary:
        return {"status": "already_ready"}
    await tool_fns.end_conversation(
        session_id=session_id,
        user_id=session.user_id if session else None,
        transcript=req.transcript,
    )
    return {"status": "summary_ready"}


@router.get("/sessions")
async def list_sessions(limit: int = 25):
    db = await get_db()
    sessions = await queries.list_sessions(db, max(1, min(limit, 100)))

    return {
        "sessions": [
            {
                "id": session.id,
                "session_id": session.session_id,
                "user_id": session.user_id,
                "transcript": _json_or_none(session.transcript),
                "summary": _json_or_none(session.summary),
                "cost_breakdown": _json_or_none(session.cost_breakdown),
                "started_at": session.started_at,
                "ended_at": session.ended_at,
            }
            for session in sessions
        ]
    }


def _tool_display(tool: str, result: dict) -> str:
    if tool == "identify_user":
        return "Patient identified"
    if tool == "set_user_name":
        return "Name saved"
    if tool == "fetch_slots":
        return f"{result.get('count', 0)} slots available"
    if tool == "book_appointment":
        if result.get("success"):
            return f"Booking confirmed - {result.get('date')} at {result.get('time_slot')}"
        return f"Booking failed: {result.get('reason', 'unknown error')}"
    if tool == "retrieve_appointments":
        return f"{result.get('count', 0)} appointment(s) found"
    if tool == "cancel_appointment":
        return "Appointment cancelled" if result.get("success") else result.get("reason", "Cancellation failed")
    if tool == "modify_appointment":
        if result.get("success"):
            return f"Rescheduled to {result.get('date')} at {result.get('time_slot')}"
        return result.get("reason", "Modification failed")
    if tool == "end_conversation":
        return "Summary ready"
    return tool
