import json
from fastapi import APIRouter, HTTPException
from db.database import get_db
from db import queries

router = APIRouter()


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
        "started_at": session.started_at,
        "ended_at": session.ended_at,
    }
