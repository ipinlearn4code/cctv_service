import requests
import logging
import os
from datetime import datetime

def upload_file(file_path: str, file_type: str, cctv_id: str, endpoint: str, detection_type: str = "unknown"):
    """
    Upload file to external API according to upload_endpoints.md specification
    
    Args:
        file_path: Path to the file to upload
        file_type: Type of file (video, screenshot)
        cctv_id: CCTV camera ID
        endpoint: External endpoint URL
        detection_type: Type of detection (weapon, crowd, weapon_crowd)
    """
    try:
        # Get CCTV name from config or use ID
        try:
            from app.core.csv_manager import read_csv
            df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
            df["id"] = df["id"].astype(str).str.strip().str.strip('"')
            cctv_row = df[df["id"] == str(cctv_id).strip().strip('"')]
            cctv_name = cctv_row.iloc[0]["name"] if not cctv_row.empty else f"camera_{cctv_id}"
        except Exception as e:
            logging.warning(f"Could not get CCTV name for {cctv_id}: {e}")
            cctv_name = f"camera_{cctv_id}"
        
        # Prepare the upload according to API specification
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, "application/octet-stream")
            }
            data = {
                "cctv_name": cctv_name,
                "detection_type": detection_type,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            # Construct the full upload endpoint
            upload_url = endpoint.rstrip('/') + '/api/cctv/upload'
            
            response = requests.post(upload_url, files=files, data=data, timeout=30)
            response.raise_for_status()
            
            # Log successful upload
            response_data = response.json() if response.content else {}
            if response_data.get("success"):
                storage_path = response_data.get("data", {}).get("storage_path", "unknown")
                logging.info(f"Successfully uploaded {file_type} for CCTV {cctv_id} ({cctv_name}) to {storage_path}")
            else:
                logging.warning(f"Upload response indicates failure: {response_data}")
                
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error uploading {file_type} for CCTV {cctv_id}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error uploading {file_type} for CCTV {cctv_id}: {e}")
        raise