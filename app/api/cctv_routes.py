# File: app/api/cctv_routes.py
from fastapi import APIRouter, HTTPException
from uuid import UUID, uuid4
from app.schemas.cctv import CCTVCreate, CCTVUpdate, CCTVResponse
from app.core.csv_manager import read_csv, write_csv
import pandas as pd

router = APIRouter()

CSV_FILE = "data/cctv_config.csv"

@router.post("", response_model=CCTVResponse)
async def create_cctv(cctv: CCTVCreate):
    df = read_csv(CSV_FILE, ["id", "name", "ip_address", "location", "status"])
    cctv_dict = cctv.dict()
    cctv_dict["id"] = str(uuid4())
    df = df.append(cctv_dict, ignore_index=True)
    write_csv(CSV_FILE, df)
    return cctv_dict

@router.get("/{id}", response_model=CCTVResponse)
async def get_cctv(id: UUID):
    df = read_csv(CSV_FILE, ["id", "name", "ip_address", "location", "status"])
    cctv = df[df["id"] == str(id)]
    if cctv.empty:
        raise HTTPException(status_code=404, detail="CCTV not found")
    return cctv.iloc[0].to_dict()

@router.put("/{id}", response_model=CCTVResponse)
async def update_cctv(id: UUID, cctv: CCTVUpdate):
    df = read_csv(CSV_FILE, ["id", "name", "ip_address", "location", "status"])
    if str(id) not in df["id"].values:
        raise HTTPException(status_code=404, detail="CCTV not found")
    update_data = cctv.dict(exclude_unset=True)
    for key, value in update_data.items():
        df.loc[df["id"] == str(id), key] = value
    write_csv(CSV_FILE, df)
    return df[df["id"] == str(id)].iloc[0].to_dict()

@router.delete("/{id}")
async def delete_cctv(id: UUID):
    df = read_csv(CSV_FILE, ["id", "name", "ip_address", "location", "status"])
    if str(id) not in df["id"].values:
        raise HTTPException(status_code=404, detail="CCTV not found")
    df = df[df["id"] != str(id)]
    write_csv(CSV_FILE, df)
    return {"message": "CCTV deleted"}

@router.get("", response_model=list[CCTVResponse])
async def list_cctv():
    df = read_csv(CSV_FILE, ["id", "name", "ip_address", "location", "status"])
    return df.to_dict(orient="records")