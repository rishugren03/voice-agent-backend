"""
All SQL lives here. No query strings anywhere else in the codebase.
Every function that returns a list uses a single query — no loops with queries inside.
"""
import aiosqlite
from typing import Optional
from .models import Appointment, CallSession, User

MASTER_SLOTS = [
    "09:00", "09:30", "10:00", "10:30", "11:00",
    "14:00", "14:30", "15:00", "15:30",
]


# ── Users ─────────────────────────────────────────────────────────────────────

async def upsert_user(db: aiosqlite.Connection, phone: str) -> User:
    async with db.execute(
        """
        INSERT INTO users (phone)
        VALUES (?)
        ON CONFLICT(phone) DO UPDATE SET updated_at = datetime('now')
        RETURNING id, phone, name, created_at, updated_at
        """,
        [phone],
    ) as cur:
        row = await cur.fetchone()
    await db.commit()
    return User(**dict(row))


async def update_user_name(db: aiosqlite.Connection, user_id: int, name: str) -> None:
    await db.execute(
        "UPDATE users SET name = ?, updated_at = datetime('now') WHERE id = ?",
        [name, user_id],
    )
    await db.commit()


async def get_user_by_phone(db: aiosqlite.Connection, phone: str) -> Optional[User]:
    async with db.execute(
        "SELECT id, phone, name, created_at, updated_at FROM users WHERE phone = ?",
        [phone],
    ) as cur:
        row = await cur.fetchone()
    return User(**dict(row)) if row else None


# ── Appointments ──────────────────────────────────────────────────────────────

async def get_available_slots(db: aiosqlite.Connection, date: str) -> list[str]:
    async with db.execute(
        "SELECT time_slot FROM appointments WHERE date = ? AND status = 'confirmed'",
        [date],
    ) as cur:
        rows = await cur.fetchall()
    booked = {row["time_slot"] for row in rows}
    return [s for s in MASTER_SLOTS if s not in booked]


async def create_appointment(
    db: aiosqlite.Connection,
    user_id: int,
    date: str,
    time_slot: str,
    doctor: str = "Dr. Sharma",
) -> tuple[Optional[Appointment], Optional[str]]:
    """Returns (appointment, None) on success or (None, reason) on failure."""
    try:
        async with db.execute(
            """
            INSERT INTO appointments (user_id, date, time_slot, doctor)
            VALUES (?, ?, ?, ?)
            RETURNING id, user_id, date, time_slot, doctor, status, notes, created_at, updated_at
            """,
            [user_id, date, time_slot, doctor],
        ) as cur:
            row = await cur.fetchone()
        await db.commit()
        return Appointment(**dict(row)), None
    except aiosqlite.IntegrityError:
        return None, "That slot is already booked."


async def get_user_appointments(
    db: aiosqlite.Connection, user_id: int
) -> list[Appointment]:
    """Single query, no N+1. Returns all confirmed appointments for a user."""
    async with db.execute(
        """
        SELECT id, user_id, date, time_slot, doctor, status, notes, created_at, updated_at
        FROM appointments
        WHERE user_id = ? AND status = 'confirmed'
        ORDER BY date, time_slot
        """,
        [user_id],
    ) as cur:
        rows = await cur.fetchall()
    return [Appointment(**dict(r)) for r in rows]


async def cancel_appointment(
    db: aiosqlite.Connection, user_id: int, appointment_id: int
) -> bool:
    async with db.execute(
        """
        UPDATE appointments
        SET status = 'cancelled', updated_at = datetime('now')
        WHERE id = ? AND user_id = ? AND status = 'confirmed'
        """,
        [appointment_id, user_id],
    ) as cur:
        affected = cur.rowcount
    await db.commit()
    return affected > 0


async def modify_appointment(
    db: aiosqlite.Connection,
    user_id: int,
    appointment_id: int,
    new_date: str,
    new_time: str,
) -> tuple[Optional[Appointment], Optional[str]]:
    """Atomically cancels old appointment and creates new one.
    sqlite3's implicit transaction keeps both operations in the same tx.
    """
    # Step 1: cancel the old appointment (starts an implicit transaction)
    async with db.execute(
        """
        UPDATE appointments
        SET status = 'cancelled', updated_at = datetime('now')
        WHERE id = ? AND user_id = ? AND status = 'confirmed'
        """,
        [appointment_id, user_id],
    ) as cur:
        if cur.rowcount == 0:
            return None, "Appointment not found or already cancelled."

    # Step 2: insert the new one — both ops share the same implicit transaction.
    # On IntegrityError we roll back, undoing the UPDATE above too.
    try:
        async with db.execute(
            """
            INSERT INTO appointments (user_id, date, time_slot, doctor)
            SELECT ?, ?, ?, doctor FROM appointments WHERE id = ?
            RETURNING id, user_id, date, time_slot, doctor, status, notes, created_at, updated_at
            """,
            [user_id, new_date, new_time, appointment_id],
        ) as cur:
            row = await cur.fetchone()
        await db.commit()
        return Appointment(**dict(row)), None
    except aiosqlite.IntegrityError:
        await db.rollback()
        return None, "That new slot is already taken."


# ── Call Sessions ─────────────────────────────────────────────────────────────

async def create_session(db: aiosqlite.Connection, session_id: str) -> CallSession:
    async with db.execute(
        """
        INSERT INTO call_sessions (session_id)
        VALUES (?)
        ON CONFLICT(session_id) DO UPDATE SET session_id = excluded.session_id
        RETURNING id, session_id, user_id, transcript, summary, cost_breakdown, started_at, ended_at
        """,
        [session_id],
    ) as cur:
        row = await cur.fetchone()
    await db.commit()
    return CallSession(**dict(row))


async def list_sessions(
    db: aiosqlite.Connection, limit: int = 25
) -> list[CallSession]:
    async with db.execute(
        """
        SELECT 
            cs.id, cs.session_id, cs.user_id, cs.transcript, cs.summary, cs.cost_breakdown, 
            cs.started_at, cs.ended_at, u.phone AS user_phone, u.name AS user_name
        FROM call_sessions cs
        LEFT JOIN users u ON cs.user_id = u.id
        ORDER BY cs.started_at DESC, cs.id DESC
        LIMIT ?
        """,
        [limit],
    ) as cur:
        rows = await cur.fetchall()
    return [CallSession(**dict(r)) for r in rows]


async def link_session_user(
    db: aiosqlite.Connection, session_id: str, user_id: int
) -> None:
    await db.execute(
        "UPDATE call_sessions SET user_id = ? WHERE session_id = ?",
        [user_id, session_id],
    )
    await db.commit()


async def link_session_tavus_conversation(
    db: aiosqlite.Connection, session_id: str, conversation_id: str
) -> None:
    await db.execute(
        "UPDATE call_sessions SET tavus_conversation_id = ? WHERE session_id = ?",
        [conversation_id, session_id],
    )
    await db.commit()


async def save_session_summary(
    db: aiosqlite.Connection,
    session_id: str,
    transcript: str,
    summary: str,
    cost_breakdown: str,
) -> None:
    await db.execute(
        """
        UPDATE call_sessions
        SET transcript = ?, summary = ?, cost_breakdown = ?, ended_at = datetime('now')
        WHERE session_id = ?
        """,
        [transcript, summary, cost_breakdown, session_id],
    )
    await db.commit()


async def get_session(
    db: aiosqlite.Connection, session_id: str
) -> Optional[CallSession]:
    async with db.execute(
        """
        SELECT id, session_id, user_id, transcript, summary, cost_breakdown, started_at, ended_at
        FROM call_sessions WHERE session_id = ?
        """,
        [session_id],
    ) as cur:
        row = await cur.fetchone()
    return CallSession(**dict(row)) if row else None


async def get_session_by_tavus_conversation(
    db: aiosqlite.Connection, conversation_id: str
) -> Optional[CallSession]:
    async with db.execute(
        """
        SELECT id, session_id, user_id, transcript, summary, cost_breakdown, started_at, ended_at
        FROM call_sessions WHERE tavus_conversation_id = ?
        """,
        [conversation_id],
    ) as cur:
        row = await cur.fetchone()
    return CallSession(**dict(row)) if row else None


async def update_cost_breakdown(
    db: aiosqlite.Connection, session_id: str, cost_breakdown: str
) -> None:
    """Update only the cost_breakdown field — used when summary was already saved by end_conversation tool."""
    await db.execute(
        "UPDATE call_sessions SET cost_breakdown = ? WHERE session_id = ?",
        [cost_breakdown, session_id],
    )
    await db.commit()


async def get_dashboard_stats(db: aiosqlite.Connection) -> dict:
    async with db.execute(
        """
        SELECT 
            (SELECT COUNT(*) FROM call_sessions) as total_interactions,
            (SELECT COUNT(*) FROM call_sessions WHERE ended_at IS NOT NULL) as completed_sessions,
            (SELECT COUNT(DISTINCT id) FROM users) as total_leads,
            (SELECT AVG(strftime('%s', ended_at) - strftime('%s', started_at)) 
             FROM call_sessions WHERE ended_at IS NOT NULL) as avg_duration,
            (SELECT COUNT(*) FROM call_sessions WHERE started_at > datetime('now', '-1 day')) as last_24h
        """
    ) as cur:
        row = await cur.fetchone()
    
    total = row["total_interactions"] or 0
    completed = row["completed_sessions"] or 0
    completion_rate = (completed * 100.0 / total) if total > 0 else 0
    
    return {
        "total_interactions": total,
        "completion_rate": completion_rate,
        "total_leads": row["total_leads"] or 0,
        "avg_duration_seconds": row["avg_duration"] or 0,
        "last_24h_interactions": row["last_24h"] or 0
    }
