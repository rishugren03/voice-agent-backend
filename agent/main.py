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
    """
    Healthcare front-desk voice agent.
    Instance state (session_id, user_id, transcript) persists across tool calls.
    """

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

    # ── Tools ─────────────────────────────────────────────────────────────────

    @function_tool
    async def identify_user(self, _context: RunContext, phone: str) -> str:
        """Look up or create a user by phone number. Always call this first before any appointment action."""
        self._pending("identify_user", "Looking up account…")
        result = await tool_fns.identify_user(phone)
        self._user_id = result["user_id"]
        db = await get_db()
        await queries.link_session_user(db, self.session_id, self._user_id)
        return json.dumps(result)

    @function_tool
    async def set_user_name(self, _context: RunContext, user_id: int, name: str) -> str:
        """Save the user's name once you learn it from the conversation."""
        self._pending("set_user_name", "Saving name…")
        return json.dumps(await tool_fns.set_user_name(user_id, name))

    @function_tool
    async def fetch_slots(self, _context: RunContext, date: str) -> str:
        """Get available appointment slots for a given date in YYYY-MM-DD format."""
        self._pending("fetch_slots", "Fetching available slots…")
        return json.dumps(await tool_fns.fetch_slots(date))

    @function_tool
    async def book_appointment(
        self, _context: RunContext, user_id: int, date: str, time_slot: str
    ) -> str:
        """Book an appointment. Confirm date and time with the user before calling this."""
        self._pending("book_appointment", "Booking appointment…")
        return json.dumps(await tool_fns.book_appointment(user_id, date, time_slot))

    @function_tool
    async def retrieve_appointments(self, _context: RunContext, user_id: int) -> str:
        """Get all confirmed appointments for a user."""
        self._pending("retrieve_appointments", "Loading appointments…")
        return json.dumps(await tool_fns.retrieve_appointments(user_id))

    @function_tool
    async def cancel_appointment(
        self, _context: RunContext, user_id: int, appointment_id: int
    ) -> str:
        """Cancel a specific appointment by its ID."""
        self._pending("cancel_appointment", "Cancelling appointment…")
        return json.dumps(await tool_fns.cancel_appointment(user_id, appointment_id))

    @function_tool
    async def modify_appointment(
        self,
        _context: RunContext,
        user_id: int,
        appointment_id: int,
        new_date: str,
        new_time: str,
    ) -> str:
        """Change an existing appointment to a new date and time slot."""
        self._pending("modify_appointment", "Updating appointment…")
        return json.dumps(
            await tool_fns.modify_appointment(user_id, appointment_id, new_date, new_time)
        )

    @function_tool
    async def end_conversation(self, _context: RunContext) -> str:
        """End the call, generate a summary, and close the session."""
        self._pending("end_conversation", "Generating summary…")
        result = await tool_fns.end_conversation(
            self.session_id, self._user_id, self.transcript
        )
        _publish_data(self.room, {"type": "summary_ready", "session_id": self.session_id})
        return json.dumps({"status": "done", "summary": result.get("summary")})


# ── Entry point ───────────────────────────────────────────────────────────────

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
        # How long to wait after user stops speaking before responding.
        # Default is ~0.5s; 0.3s feels more natural on voice calls.
        min_endpointing_delay=0.3,
        max_endpointing_delay=0.6,
    )

    # ── Transcript collection ─────────────────────────────────────────────────
    @session.on("conversation_item_added")
    def on_item_added(event: ConversationItemAddedEvent):
        item = event.item
        if hasattr(item, "role") and hasattr(item, "text_content"):
            agent.transcript.append({
                "role": item.role,
                "content": item.text_content,  # property, not method
                "ts": _now(),
            })

    # ── Tool call status → DataChannel → frontend ─────────────────────────────
    @session.on("function_tools_executed")
    def on_tools_executed(event: FunctionToolsExecutedEvent):
        for call, _output in event.zipped():
            _publish_data(ctx.room, {
                "type": "tool_call",
                "tool": call.name,
                "status": "done",
                "display": _tool_display(call.name),
            })

    await session.start(agent=agent, room=ctx.room)

    # Wait for the frontend participant to join, then give the browser a moment
    # to unlock audio (room.startAudio / el.play) before sending the greeting.
    # Without the delay the greeting audio arrives before autoplay is unblocked.
    await ctx.wait_for_participant()
    await asyncio.sleep(1.5)
    await session.say("Hello! I'm Maya, your Mykare healthcare assistant. How can I help you today?")

    await session.wait_for_inactive()
    await _finalize(agent)


async def _finalize(agent: MayaAgent) -> None:
    if agent._finalized:
        return
    agent._finalized = True

    db = await get_db()
    summary_result = await tool_fns.end_conversation(
        agent.session_id, agent._user_id, agent.transcript
    )
    await queries.save_session_summary(
        db,
        session_id=agent.session_id,
        transcript=json.dumps(agent.transcript),
        summary=json.dumps(summary_result.get("summary", {})),
        cost_breakdown=json.dumps(agent.cost.to_dict()),
    )
    logger.info(f"Session {agent.session_id} saved. Cost: {agent.cost.to_dict()['total_usd']} USD")


def _publish_data(room: rtc.Room, payload: dict) -> None:
    asyncio.ensure_future(
        room.local_participant.publish_data(
            json.dumps(payload).encode(), reliable=True
        )
    )


def _tool_display(tool_name: str) -> str:
    return {
        "identify_user":         "Account found",
        "set_user_name":         "Name saved",
        "fetch_slots":           "Slots loaded",
        "book_appointment":      "Booking confirmed ✅",
        "retrieve_appointments": "Appointments loaded",
        "cancel_appointment":    "Appointment cancelled",
        "modify_appointment":    "Appointment updated ✅",
        "end_conversation":      "Summary ready",
    }.get(tool_name, tool_name)


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
