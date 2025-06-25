from fastapi import APIRouter, HTTPException
from app.schemas.detection_config import DetectionConfig
from app.core.csv_manager import read_csv, write_csv
import pandas as pd

router = APIRouter()

CSV_FILE = "data/detection_config.csv"

@router.get("", response_model=DetectionConfig)
async def get_detection_config():
    df = read_csv(CSV_FILE, ["record_duration", "enable_video", "enable_screenshot", "external_endpoint", "jwt_secret"])
    if df.empty:
        raise HTTPException(status_code=404, detail="Detection config not found")
    return df.iloc[0].to_dict()

@router.put("", response_model=DetectionConfig)
async def update_detection_config(config: DetectionConfig):
    config_dict = config.dict()
    df = pd.DataFrame([config_dict])
    write_csv(CSV_FILE, df)
    return config_dict