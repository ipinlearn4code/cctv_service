import requests
import logging

def upload_file(file_path: str, file_type: str, cctv_id: str, endpoint: str):
    with open(file_path, "rb") as f:
        files = {file_type: (file_path.split("/")[-1], f)}
        data = {"cctv_id": cctv_id}
        response = requests.post(endpoint, files=files, data=data)
        response.raise_for_status()
        logging.info(f"Uploaded {file_type} for CCTV {cctv_id} to {endpoint}")