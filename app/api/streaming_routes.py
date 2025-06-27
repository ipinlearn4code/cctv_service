from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, List
from app.services.stream_processor import generate_frames, processor
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

@router.post("/processor/{cctv_id}/start")
async def start_processing(cctv_id: str):
    """Start background processing for a specific CCTV camera"""
    try:
        # Check if CCTV exists
        df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
        df["id"] = df["id"].astype(str).str.strip().str.strip('"')
        cctv_id_clean = str(cctv_id).strip().strip('"')
        
        if cctv_id_clean not in df["id"].values:
            raise HTTPException(status_code=404, detail=f"CCTV with id '{cctv_id_clean}' not found")
        
        # Start background processing
        processor.start_background_processing(cctv_id_clean)
        return {"message": f"Started background processing for CCTV {cctv_id_clean}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/processor/{cctv_id}/stop")
async def stop_processing(cctv_id: str):
    """Stop background processing for a specific CCTV camera"""
    try:
        cctv_id_clean = str(cctv_id).strip().strip('"')
        processor.stop_background_processing(cctv_id_clean)
        return {"message": f"Stopped background processing for CCTV {cctv_id_clean}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processor/status")
async def get_processor_status():
    """Get status of all background processors"""
    try:
        active_cams = list(processor.active_cams.keys())
        return {
            "active_cameras": active_cams,
            "total_active": len(active_cams)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
