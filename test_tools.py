"""
Phase 3 test — run from backend/ with: python test_tools.py
Tests all tool functions directly without LiveKit or a real voice call.
OPENAI_API_KEY must be set (used only by end_conversation).
"""
import asyncio
import os
import json

os.environ["DB_PATH"] = "mykare_tools_test.db"

# Remove stale test DB from previous interrupted runs
for _f in ["mykare_tools_test.db", "mykare_tools_test.db-wal", "mykare_tools_test.db-shm"]:
    if os.path.exists(_f):
        os.remove(_f)

from db.database import init_db, close_db
from agent import tools


async def main():
    await init_db()
    print("✓ DB ready\n")

    # 1. identify_user
    result = await tools.identify_user("9876543210")
    assert "user_id" in result and result["phone"] == "9876543210"
    user_id = result["user_id"]
    print(f"✓ identify_user   : user_id={user_id}  is_new={result['is_new']}")

    # 2. set_user_name
    result = await tools.set_user_name(user_id, "Rishu")
    assert result["success"]
    print(f"✓ set_user_name   : name saved")

    # 3. fetch_slots
    result = await tools.fetch_slots("2026-05-10")
    assert result["count"] == 9
    print(f"✓ fetch_slots     : {result['count']} slots → {result['available_slots']}")

    # 4. book_appointment
    result = await tools.book_appointment(user_id, "2026-05-10", "10:00")
    assert result["success"]
    appt_id = result["appointment_id"]
    print(f"✓ book_appointment: id={appt_id}  {result['date']} {result['time_slot']}")

    # 5. double-book rejected
    result = await tools.book_appointment(user_id, "2026-05-10", "10:00")
    assert not result["success"]
    print(f"✓ double-book blocked: '{result['reason']}'")

    # 6. retrieve_appointments
    result = await tools.retrieve_appointments(user_id)
    assert result["count"] == 1
    print(f"✓ retrieve_appointments: {result['count']} appointment(s)")

    # 7. modify_appointment
    result = await tools.modify_appointment(user_id, appt_id, "2026-05-10", "09:00")
    assert result["success"]
    new_appt_id = result["new_appointment_id"]
    print(f"✓ modify_appointment: moved to {result['date']} {result['time_slot']}  new id={new_appt_id}")

    # 8. cancel_appointment
    result = await tools.cancel_appointment(user_id, new_appt_id)
    assert result["success"]
    print(f"✓ cancel_appointment: cancelled id={new_appt_id}")

    # 9. end_conversation (calls OpenAI — skipped if no key)
    if os.getenv("OPENAI_API_KEY"):
        transcript = [
            {"role": "assistant", "content": "Hello! I'm Maya. How can I help?"},
            {"role": "user", "content": "I want to book an appointment for tomorrow at 10am"},
            {"role": "assistant", "content": "Sure, may I get your phone number?"},
            {"role": "user", "content": "It's 9876543210"},
        ]
        from db import queries
        from db.database import get_db
        db = await get_db()
        await queries.create_session(db, "test-session-tools-001")
        await queries.link_session_user(db, "test-session-tools-001", user_id)

        result = await tools.end_conversation("test-session-tools-001", user_id, transcript)
        assert "summary" in result
        print(f"✓ end_conversation: summary generated → {json.dumps(result['summary'], indent=2)}")
    else:
        print("⚠ end_conversation skipped (no OPENAI_API_KEY)")

    await close_db()

    for f in ["mykare_tools_test.db", "mykare_tools_test.db-wal", "mykare_tools_test.db-shm"]:
        if os.path.exists(f):
            os.remove(f)

    print("\n✅  All Phase 3 tests passed!")


asyncio.run(main())
