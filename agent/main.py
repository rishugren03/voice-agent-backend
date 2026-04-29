"""
LiveKit voice agent entry point.
Run from backend/ with: python -m agent.main dev
"""
import asyncio
import json
import logging
import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
)
from livekit.plugins import deepgram, cartesia, silero
from livekit.plugins import openai as lk_openai

from db.database import init_db, get_db
from db import queries
from agent import tools as tool_fns
from agent.prompts import SYSTEM_PROMPT
from agent.cost_tracker import CostTracker

logger = logging.getLogger("mykare-agent")


class MayaAgent(Agent):
    """Healthcare front-desk voice agent."""

    def __init__(self, session_id: str, room: rtc.Room):
        super().__init__(instructions=SYSTEM_PROMPT)
        self.session_id = session_id
        self.room = room
        self.cost = CostTracker()
        self.transcript: list[dict] = []
        self._user_id: Optional[int] = None
        self._finalized = False

    def _pending(self, tool: str, display: str) -> None:
        _publish_data(self.room, {"type": "tool_call", "tool": tool, "status": "pending", "display": display})

    def _done(self, tool: str, display: str) -> None:
        _publish_data(self.room, {"type": "tool_call", "tool": tool, "status": "done", "display": display})

    # ── Tools ──────────────────────────────────────────────────────────────────

    @function_tool
    async def identify_user(self, _context: RunContext, phone: str) -> str:
        """Look up or create a user by phone number. Always call this first before any appointment action."""
        self._pending("identify_user", "Looking up account…")
        result = await tool_fns.identify_user(phone)
        self._user_id = result["user_id"]
        db = await get_db()
        await queries.link_session_user(db, self.session_id, self._user_id)
        self._done("identify_user", "Account found ✓")
        return json.dumps(result)

    @function_tool
    async def set_user_name(self, _context: RunContext, name: str) -> str:
        """Save the user's name once you learn it from the conversation."""
        self._pending("set_user_name", "Saving name…")
        if not self._user_id:
            return json.dumps({"success": False, "reason": "identify_user must be called first."})
        result = await tool_fns.set_user_name(self._user_id, name)
        self._done("set_user_name", "Name saved ✓")
        return json.dumps(result)

    @function_tool
    async def fetch_slots(self, _context: RunContext, date: str) -> str:
        """Get available appointment slots for a given date in YYYY-MM-DD format."""
        self._pending("fetch_slots", "Fetching available slots…")
        result = await tool_fns.fetch_slots(date)
        self._done("fetch_slots", f"{result.get('count', 0)} slots available")
        return json.dumps(result)

    @function_tool
    async def book_appointment(self, _context: RunContext, date: str, time_slot: str) -> str:
        """Book an appointment. Confirm date and time with the user before calling this."""
        self._pending("book_appointment", "Booking appointment…")
        if not self._user_id:
            return json.dumps({"success": False, "reason": "identify_user must be called first."})
        result = await tool_fns.book_appointment(self._user_id, date, time_slot)
        if result.get("success"):
            self._done("book_appointment", f"Booking confirmed ✅ — {date} at {time_slot}")
        else:
            self._done("book_appointment", f"Booking failed: {result.get('reason', 'unknown error')}")
        return json.dumps(result)

    @function_tool
    async def retrieve_appointments(self, _context: RunContext) -> str:
        """Get all confirmed appointments for the current user."""
        self._pending("retrieve_appointments", "Loading appointments…")
        if not self._user_id:
            return json.dumps({"success": False, "reason": "identify_user must be called first."})
        result = await tool_fns.retrieve_appointments(self._user_id)
        self._done("retrieve_appointments", f"{result.get('count', 0)} appointment(s) found")
        return json.dumps(result)

    @function_tool
    async def cancel_appointment(self, _context: RunContext, appointment_id: int) -> str:
        """Cancel a specific appointment by its ID."""
        self._pending("cancel_appointment", "Cancelling appointment…")
        if not self._user_id:
            return json.dumps({"success": False, "reason": "identify_user must be called first."})
        result = await tool_fns.cancel_appointment(self._user_id, appointment_id)
        if result.get("success"):
            self._done("cancel_appointment", "Appointment cancelled ✓")
        else:
            self._done("cancel_appointment", f"Cancellation failed: {result.get('reason', '')}")
        return json.dumps(result)

    @function_tool
    async def modify_appointment(
        self,
        _context: RunContext,
        appointment_id: int,
        new_date: str,
        new_time: str,
    ) -> str:
        """Change an existing appointment to a new date and time slot."""
        self._pending("modify_appointment", "Updating appointment…")
        if not self._user_id:
            return json.dumps({"success": False, "reason": "identify_user must be called first."})
        result = await tool_fns.modify_appointment(self._user_id, appointment_id, new_date, new_time)
        if result.get("success"):
            self._done("modify_appointment", f"Rescheduled to {new_date} at {new_time} ✅")
        else:
            self._done("modify_appointment", f"Modification failed: {result.get('reason', '')}")
        return json.dumps(result)

    @function_tool
    async def end_conversation(self, _context: RunContext) -> str:
        """End the call, generate a summary, and close the session."""
        self._pending("end_conversation", "Generating summary…")
        self._finalized = True  # prevent _finalize from double-running

        summary = await tool_fns.generate_summary(self.transcript)
        db = await get_db()
        await queries.save_session_summary(
            db,
            session_id=self.session_id,
            transcript=json.dumps(self.transcript),
            summary=json.dumps(summary),
            cost_breakdown=json.dumps(self.cost.to_dict()),
        )

        self._done("end_conversation", "Summary ready 📋")
        # Notify frontend AFTER summary is safely in DB
        _publish_data(self.room, {"type": "summary_ready", "session_id": self.session_id})
        return json.dumps({"status": "done"})


# ── Entry point ────────────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext):
    await init_db()
    await ctx.connect()

    session_id = ctx.room.name
    db = await get_db()
    await queries.create_session(db, session_id)
    logger.info(f"Session started: {session_id}")

    agent = MayaAgent(session_id=session_id, room=ctx.room)

    session = AgentSession(
        stt=deepgram.STT(),
        llm=lk_openai.LLM(model="gpt-4o-2024-11-20"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        min_endpointing_delay=0.3,
        max_endpointing_delay=0.6,
    )

    # ── Transcript collection ──────────────────────────────────────────────────
    @session.on("conversation_item_added")
    def on_item_added(event: ConversationItemAddedEvent):
        item = event.item
        role = getattr(item, "role", None)
        if role not in ("user", "assistant"):
            return
        content = getattr(item, "text_content", None)
        if content:
            agent.transcript.append({"role": role, "content": content, "ts": _now()})

    # ── Tool call "done" → DataChannel → frontend ──────────────────────────────
    # NOTE: _pending() is called inside each tool method for immediate feedback.
    # _done() is also called inside each tool method for accurate result messages.
    # The function_tools_executed event is a fallback for any tools that miss it.
    @session.on("function_tools_executed")
    def on_tools_executed(event: FunctionToolsExecutedEvent):
        pass  # done signals are sent directly from each tool method above

    await session.start(agent=agent, room=ctx.room)

    await ctx.wait_for_participant()
    await asyncio.sleep(1.5)
    await session.say("Hello! I'm Maya, your Mykare healthcare assistant. How can I help you today?")

    await session.wait_for_inactive()
    await _finalize(agent)


async def _finalize(agent: MayaAgent) -> None:
    """Called when the session ends. Saves summary + notifies frontend."""
    db = await get_db()

    if agent._finalized:
        # end_conversation tool already saved the summary.
        # Just update the cost breakdown (it was $0 when tool ran).
        await queries.update_cost_breakdown(db, agent.session_id, json.dumps(agent.cost.to_dict()))
        logger.info(f"Session {agent.session_id} cost updated: {agent.cost.to_dict()['total_usd']} USD")
        return

    agent._finalized = True
    logger.info(f"Generating summary for {agent.session_id} (agent did not call end_conversation)")

    summary = await tool_fns.generate_summary(agent.transcript)
    await queries.save_session_summary(
        db,
        session_id=agent.session_id,
        transcript=json.dumps(agent.transcript),
        summary=json.dumps(summary),
        cost_breakdown=json.dumps(agent.cost.to_dict()),
    )
    # This is the primary path where summary_ready fires when user just hangs up.
    _publish_data(agent.room, {"type": "summary_ready", "session_id": agent.session_id})
    logger.info(f"Session {agent.session_id} saved. Cost: {agent.cost.to_dict()['total_usd']} USD")


def _publish_data(room: rtc.Room, payload: dict) -> None:
    async def _send():
        try:
            await room.local_participant.publish_data(
                json.dumps(payload).encode(), reliable=True
            )
        except Exception as e:
            logger.warning(f"publish_data failed ({payload.get('type')}): {e}")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_send())
    except Exception as e:
        logger.warning(f"_publish_data scheduling failed: {e}")


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
