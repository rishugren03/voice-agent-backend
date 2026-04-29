"""
Phase 1 test — run from backend/ with: python test_db.py
Tests: upsert, slot availability, booking, double-book prevention,
       retrieval (no N+1), cancellation, re-booking, modification, sessions.
"""
import asyncio
import json
import os

os.environ["DB_PATH"] = "mykare_test.db"  # use a throwaway DB

from db.database import init_db, get_db, close_db
from db import queries

PHONE = "9876543210"
DATE = "2026-05-10"


async def main():
    await init_db()
    db = await get_db()
    print("✓ DB initialized\n")

    # ── Users ──────────────────────────────────────────────────────────────────
    user = await queries.upsert_user(db, PHONE)
    print(f"✓ User created   : id={user.id}  phone={user.phone}")

    same_user = await queries.upsert_user(db, PHONE)
    assert same_user.id == user.id, "FAIL: upsert created a duplicate"
    print(f"✓ Upsert is idempotent (same id={user.id})")

    await queries.update_user_name(db, user.id, "Rishu")
    named_user = await queries.get_user_by_phone(db, PHONE)
    assert named_user.name == "Rishu"
    print(f"✓ Name updated   : {named_user.name}\n")

    # ── Slots ──────────────────────────────────────────────────────────────────
    slots = await queries.get_available_slots(db, DATE)
    assert len(slots) == 9, f"Expected 9 slots, got {len(slots)}"
    print(f"✓ All 9 slots available for {DATE}")

    # ── Booking ────────────────────────────────────────────────────────────────
    appt, err = await queries.create_appointment(db, user.id, DATE, "10:00")
    assert err is None and appt is not None
    print(f"✓ Appointment booked : id={appt.id}  {appt.date} {appt.time_slot}")

    # ── Double-book prevention ─────────────────────────────────────────────────
    appt2, err2 = await queries.create_appointment(db, user.id, DATE, "10:00")
    assert appt2 is None and err2 is not None
    print(f"✓ Double-book blocked: '{err2}'")

    slots_after = await queries.get_available_slots(db, DATE)
    assert "10:00" not in slots_after and len(slots_after) == 8
    print(f"✓ Available slots reduced to {len(slots_after)}\n")

    # ── Retrieval (single query, no N+1) ───────────────────────────────────────
    user_appts = await queries.get_user_appointments(db, user.id)
    assert len(user_appts) == 1
    print(f"✓ Retrieval returned {len(user_appts)} appointment(s)")

    # ── Cancellation ───────────────────────────────────────────────────────────
    ok = await queries.cancel_appointment(db, user.id, appt.id)
    assert ok
    print(f"✓ Appointment {appt.id} cancelled")

    bad_cancel = await queries.cancel_appointment(db, user.id, appt.id)
    assert not bad_cancel
    print(f"✓ Re-cancel correctly returned False")

    # ── Re-book a cancelled slot ───────────────────────────────────────────────
    appt3, err3 = await queries.create_appointment(db, user.id, DATE, "10:00")
    assert err3 is None and appt3 is not None
    print(f"✓ Cancelled slot re-booked : id={appt3.id}\n")

    # ── Modify (atomic cancel + rebook) ────────────────────────────────────────
    new_appt, err4 = await queries.modify_appointment(db, user.id, appt3.id, DATE, "09:00")
    assert err4 is None and new_appt is not None
    print(f"✓ Modified to {new_appt.date} {new_appt.time_slot}  new id={new_appt.id}")

    # Modifying to a taken slot should fail
    await queries.create_appointment(db, user.id, DATE, "09:30")
    _, err5 = await queries.modify_appointment(db, user.id, new_appt.id, DATE, "09:30")
    assert err5 is not None
    print(f"✓ Modify to taken slot blocked: '{err5}'\n")

    # ── Session lifecycle ──────────────────────────────────────────────────────
    session = await queries.create_session(db, "test-session-001")
    print(f"✓ Session created  : {session.session_id}")

    await queries.link_session_user(db, session.session_id, user.id)
    await queries.save_session_summary(
        db,
        session_id=session.session_id,
        transcript=json.dumps([{"role": "user", "content": "Hello", "ts": "2026-05-10T09:00:00Z"}]),
        summary=json.dumps({"overview": "Test call", "appointments": [], "extracted": {}}),
        cost_breakdown=json.dumps({"stt_usd": 0.001, "tts_usd": 0.002, "llm_usd": 0.005}),
    )

    fetched = await queries.get_session(db, session.session_id)
    assert fetched.user_id == user.id
    assert fetched.ended_at is not None
    print(f"✓ Session saved & retrieved (ended_at={fetched.ended_at})")

    await close_db()

    # Cleanup test DB
    if os.path.exists("mykare_test.db"):
        os.remove("mykare_test.db")
    if os.path.exists("mykare_test.db-wal"):
        os.remove("mykare_test.db-wal")
    if os.path.exists("mykare_test.db-shm"):
        os.remove("mykare_test.db-shm")

    print("\n✅  All Phase 1 tests passed!")


asyncio.run(main())
