from fastapi import APIRouter
from db.database import get_db
from db import queries

router = APIRouter()

@router.get("/stats")
async def get_dashboard_stats():
    db = await get_db()
    return await queries.get_dashboard_stats(db)
