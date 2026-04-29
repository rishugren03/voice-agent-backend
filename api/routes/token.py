import os
import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from livekit.api import AccessToken, VideoGrants

router = APIRouter()


class TokenRequest(BaseModel):
    room: str
    identity: str   # use phone number as identity


class TokenResponse(BaseModel):
    token: str
    url: str


@router.post("/token", response_model=TokenResponse)
async def create_token(req: TokenRequest):
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    livekit_url = os.getenv("LIVEKIT_URL")

    missing = [k for k, v in {
        "LIVEKIT_API_KEY": api_key,
        "LIVEKIT_API_SECRET": api_secret,
        "LIVEKIT_URL": livekit_url,
    }.items() if not v]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env vars: {missing}")

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(req.identity)
        .with_name(req.identity)
        .with_grants(VideoGrants(room_join=True, room=req.room))
        .with_ttl(datetime.timedelta(hours=1))
        .to_jwt()
    )

    return TokenResponse(token=token, url=livekit_url)
