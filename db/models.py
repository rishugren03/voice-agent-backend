from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class User:
    id: int
    phone: str
    name: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class Appointment:
    id: int
    user_id: int
    date: str        # YYYY-MM-DD
    time_slot: str   # HH:MM (24h)
    doctor: str
    status: str      # 'confirmed' | 'cancelled'
    notes: Optional[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CallSession:
    id: int
    session_id: str
    user_id: Optional[int]
    transcript: Optional[str]      # JSON string: [{role, content, ts}]
    summary: Optional[str]         # JSON string: see summary schema in plan
    cost_breakdown: Optional[str]  # JSON string: {stt_usd, tts_usd, llm_usd}
    started_at: str
    ended_at: Optional[str]
    user_phone: Optional[str] = None
    user_name: Optional[str] = None
