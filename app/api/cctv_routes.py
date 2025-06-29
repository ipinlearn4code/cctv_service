# File: app/api/cctv_routes.py
from fastapi import APIRouter, HTTPException
from uuid import UUID, uuid4
from app.schemas.cctv import CCTVCreate, CCTVUpdate, CCTVResponse
from app.core.csv_manager import read_csv, write_csv
from app.services.cctv_monitor import cctv_monitor
import pandas as pd
import logging

router = APIRouter()

CSV_FILE = "data/cctv_config.csv"

@router.post("", response_model=CCTVResponse)
async def create_cctv(cctv: CCTVCreate):
    df = read_csv(CSV_FILE, ["id", "name", "ip_address", "location", "status"])
    cctv_dict = cctv.dict()
    cctv_dict["id"] = str(uuid4())
    # Ensure all values are strings
    for key in cctv_dict:
        cctv_dict[key] = str(cctv_dict[key])
    
    # Use pd.concat instead of deprecated append
    new_row = pd.DataFrame([cctv_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    write_csv(CSV_FILE, df)
    
    # If the new camera is set to active, trigger immediate check
    if str(cctv_dict["status"]).lower() == "active":
        try:
            cctv_monitor.force_check_camera(cctv_dict["id"])
            logging.info(f"Triggered immediate connection check for new CCTV {cctv_dict['id']}")
        except Exception as e:
            logging.warning(f"Failed to trigger immediate check for new CCTV {cctv_dict['id']}: {e}")
    
    return cctv_dict

@router.get("/{id}", response_model=CCTVResponse)
async def get_cctv(id: str):
    df = read_csv(CSV_FILE, ["id", "name", "ip_address", "location", "status"])
    df["id"] = df["id"].astype(str)  # Ensure ID is string
    cctv = df[df["id"] == str(id)]
    if cctv.empty:
        raise HTTPException(status_code=404, detail="CCTV not found")
    result = cctv.iloc[0].to_dict()
    # Ensure all fields are strings
    for key in result:
        result[key] = str(result[key])
    return result

@router.put("/{id}", response_model=CCTVResponse)
async def update_cctv(id: str, cctv: CCTVUpdate):
    df = read_csv(CSV_FILE, ["id", "name", "ip_address", "location", "status"])
    df["id"] = df["id"].astype(str)  # Ensure ID is string
    if str(id) not in df["id"].values:
        raise HTTPException(status_code=404, detail="CCTV not found")
    
    update_data = cctv.dict(exclude_unset=True)
    old_status = None
    new_status = None
    
    # Track status changes for monitoring integration
    if "status" in update_data:
        old_row = df[df["id"] == str(id)].iloc[0]
        old_status = str(old_row["status"]).lower()
        new_status = str(update_data["status"]).lower()
    
    for key, value in update_data.items():
        df.loc[df["id"] == str(id), key] = value
    
    write_csv(CSV_FILE, df)
    
    # Log status changes and trigger immediate check if needed
    if old_status and new_status and old_status != new_status:
        logging.info(f"CCTV {id} status manually changed: {old_status} -> {new_status}")
        
        # If changed to active or disconnect, trigger immediate check
        if new_status in ["active", "disconnect"]:
            try:
                cctv_monitor.force_check_camera(str(id))
                logging.info(f"Triggered immediate connection check for CCTV {id}")
            except Exception as e:
                logging.warning(f"Failed to trigger immediate check for CCTV {id}: {e}")
    
    result = df[df["id"] == str(id)].iloc[0].to_dict()
    # Ensure all fields are strings
    for key in result:
        result[key] = str(result[key])
    return result

@router.delete("/{id}")
async def delete_cctv(id: str):
    df = read_csv(CSV_FILE, ["id", "name", "ip_address", "location", "status"])
    df["id"] = df["id"].astype(str)  # Ensure ID is string
    if str(id) not in df["id"].values:
        raise HTTPException(status_code=404, detail="CCTV not found")
    df = df[df["id"] != str(id)]
    write_csv(CSV_FILE, df)
    return {"message": "CCTV deleted"}

@router.get("", response_model=list[CCTVResponse])
async def list_cctv():
    df = read_csv(CSV_FILE, ["id", "name", "ip_address", "location", "status"])
    # Ensure all fields are strings to match the response model
    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].astype(str)
    df["ip_address"] = df["ip_address"].astype(str)
    df["location"] = df["location"].astype(str)
    df["status"] = df["status"].astype(str)
    return df.to_dict(orient="records")