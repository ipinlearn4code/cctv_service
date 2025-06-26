from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from uuid import UUID
from app.services.stream_processor import generate_frames
from app.core.csv_manager import read_csv

router = APIRouter()

@router.get("/{cctv_id}")
async def stream(cctv_id: str):
    df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
    if cctv_id not in df["id"].astype(str).values:
        raise HTTPException(status_code=404, detail=f"CCTV id '{cctv_id}' not found")
    try:
        return StreamingResponse(
            generate_frames(cctv_id),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Streaming error: " + str(e))
