from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from db.database import init_db, close_db
from api.routes.token import router as token_router
from api.routes.session import router as session_router
from api.routes.tavus import router as tavus_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(title="Mykare Voice AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten before production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(token_router, prefix="/api")
app.include_router(session_router, prefix="/api")
app.include_router(tavus_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
