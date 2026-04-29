import os
import aiosqlite
from typing import Optional

DB_PATH = os.getenv("DB_PATH", "mykare.db")

# Partial unique index on (date, time_slot) WHERE status='confirmed' is the
# double-booking guard. It allows the same slot to exist as 'cancelled' so
# a slot can be re-booked after cancellation.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    phone       TEXT    UNIQUE NOT NULL,
    name        TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS appointments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date        TEXT    NOT NULL,
    time_slot   TEXT    NOT NULL,
    doctor      TEXT    NOT NULL DEFAULT 'Dr. Sharma',
    status      TEXT    NOT NULL DEFAULT 'confirmed'
                        CHECK (status IN ('confirmed', 'cancelled')),
    notes       TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_appt_no_double_book
    ON appointments(date, time_slot) WHERE status = 'confirmed';

CREATE INDEX IF NOT EXISTS idx_appt_user
    ON appointments(user_id);

CREATE INDEX IF NOT EXISTS idx_appt_date_status
    ON appointments(date, status);

CREATE TABLE IF NOT EXISTS call_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    UNIQUE NOT NULL,
    user_id         INTEGER REFERENCES users(id),
    transcript      TEXT,
    summary         TEXT,
    cost_breakdown  TEXT,
    started_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_session_user
    ON call_sessions(user_id);
"""

_db: Optional[aiosqlite.Connection] = None


async def get_db() -> aiosqlite.Connection:
    global _db
    if _db is None:
        _db = await aiosqlite.connect(DB_PATH, check_same_thread=False)
        _db.row_factory = aiosqlite.Row
        await _db.execute("PRAGMA journal_mode=WAL")
        await _db.execute("PRAGMA foreign_keys=ON")
    return _db


async def init_db() -> None:
    db = await get_db()
    await db.executescript(_SCHEMA)
    await db.commit()


async def close_db() -> None:
    global _db
    if _db is not None:
        await _db.close()
        _db = None
