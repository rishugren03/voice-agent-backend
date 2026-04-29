# Mykare AI — Backend

FastAPI server and optional LiveKit voice agent for the Mykare appointment assistant. Handles session lifecycle, tool execution, Tavus webhook integration, and analytics.

---

## Requirements

- Python 3.12+
- OpenAI API key (GPT-4o for call summarisation)
- Tavus API key (CVI conversation creation)

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in the required keys (see Environment Variables below)
uvicorn api.main:app --reload --port 8000
```

The SQLite database (`mykare.db`) and all tables are created automatically on first start. Migrations are applied at startup via `db/database.py`.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | GPT-4o for call summarisation |
| `TAVUS_API_KEY` | Yes | Tavus CVI conversation management |
| `TAVUS_REPLICA_ID` | No | Target replica; if unset, the first available replica is used |
| `TAVUS_PERSONA_ID` | No | Target persona; if unset, one is created automatically |
| `PUBLIC_BASE_URL` | No | Enables Tavus webhook (`/api/tavus/webhook`) for transcript delivery |
| `LIVEKIT_URL` | LiveKit only | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | LiveKit only | LiveKit API key |
| `LIVEKIT_API_SECRET` | LiveKit only | LiveKit API secret |
| `DEEPGRAM_API_KEY` | LiveKit only | Deepgram STT |
| `CARTESIA_API_KEY` | LiveKit only | Cartesia TTS |
| `DB_PATH` | No | SQLite file path (default: `mykare.db`) |

---

## File Structure

```
api/
  main.py                FastAPI app factory, CORS, lifespan hooks
  routes/
    analytics.py         GET /api/analytics/stats
    session.py           Session CRUD, tool dispatch, session finish
    tavus.py             Tavus conversation lifecycle and webhook handler
    token.py             LiveKit JWT token generation
agent/
  main.py                LiveKit agent entrypoint (MayaAgent class)
  tools.py               Tool implementations shared by agent and HTTP routes
  prompts.py             System prompt for the voice agent
  cost_tracker.py        Per-session STT/TTS/LLM cost accumulation
db/
  database.py            Connection, schema creation, migrations
  models.py              Dataclasses: User, Appointment, CallSession
  queries.py             All SQL — no query strings exist outside this file
requirements.txt
.env.example
test_db.py               Integration tests for db/queries.py
test_tools.py            Integration tests for agent/tools.py
```

---

## API Reference

### Tavus

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/tavus/conversation?session_id=` | Create a Tavus CVI conversation, return embed URL |
| `DELETE` | `/api/tavus/conversation/{id}?session_id=` | End conversation, trigger transcript sync |
| `POST` | `/api/tavus/webhook` | Receive `application.transcription_ready` events |

### Sessions

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/session/{id}/tool-call` | Execute a named tool against a session |
| `POST` | `/api/session/{id}/finish` | Generate and persist the final call summary |
| `GET` | `/api/session/{id}/summary` | Fetch session summary (202 if not ready yet) |
| `GET` | `/api/sessions?limit=25` | List recent sessions with joined user data |

### Analytics

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/analytics/stats` | Aggregate stats: total interactions, completion rate, avg duration |

### LiveKit (optional)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/token` | Issue a LiveKit JWT for room access |

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok"}` |

---

## Available Tools

These functions are in `agent/tools.py` and are called both by the LiveKit agent and by the HTTP tool-dispatch route (`/api/session/{id}/tool-call`).

| Tool | Description |
|---|---|
| `identify_user` | Upsert a user by phone number; returns `user_id` |
| `set_user_name` | Persist the patient's name once extracted from conversation |
| `fetch_slots` | Return available appointment slots for a given date |
| `book_appointment` | Book a slot; the DB unique index prevents double-booking |
| `retrieve_appointments` | Fetch all confirmed appointments for a user |
| `cancel_appointment` | Soft-cancel an appointment (sets status to `cancelled`) |
| `modify_appointment` | Atomically cancel and rebook to a new date/time |
| `generate_summary` | Call GPT-4o to produce a structured JSON summary |
| `end_conversation` | Generate summary and persist it to the session record |

---

## Database Schema

Three tables in SQLite with WAL mode and foreign key enforcement enabled:

**users** — `id`, `phone` (unique), `name`, `created_at`, `updated_at`

**appointments** — linked to `users`; `status` is `confirmed` or `cancelled`; a partial unique index on `(date, time_slot) WHERE status = 'confirmed'` prevents double-booking while allowing a slot to be rebooked after cancellation

**call_sessions** — `session_id` (unique), optional `tavus_conversation_id`, `transcript` (JSON), `summary` (JSON), `cost_breakdown` (JSON), `started_at`, `ended_at`

---

## Running the LiveKit Agent

The LiveKit agent is an alternative to Tavus for the voice pipeline.

```bash
python -m agent.main dev
```

Requires `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, and `OPENAI_API_KEY`.

---

## Running Tests

```bash
python test_db.py       # tests db/queries.py against a throwaway SQLite file
python test_tools.py    # tests agent/tools.py; skips end_conversation if OPENAI_API_KEY is unset
```

Both test files clean up after themselves.
