import os
import logging
import requests
from ultralytics import YOLO

MODEL_PATH = 'data/models/weapon/best.pt'
MODEL_URL = 'https://github.com/ipinlearn4code/weapon-detection/releases/download/v1.0/best.pt'

def download_model(url, save_path):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logging.info("Model downloaded successfully from GitHub.")
        else:
            raise Exception(f"Failed to fetch model. Status code: {response.status_code}")
    except Exception as err:
        logging.error(f"Error downloading model: {err}")
        raise

def load_weapon_model():
    """Load and return optimized weapon detection model"""
    try:
        model = YOLO(MODEL_PATH)
        model.fuse()
        logging.info("Weapon detection model loaded and optimized successfully")
        return model
    except Exception as e:
        logging.warning(f"Model not found or failed to load: {e}. Attempting to download...")
        try:
            download_model(MODEL_URL, MODEL_PATH)
            model = YOLO(MODEL_PATH)
            model.fuse()
            logging.info("Model downloaded, loaded, and optimized successfully")
            return model
        except Exception as final_error:
            logging.error(f"Failed to download or load weapon model: {final_error}")
            logging.error(f"Cannot proceed without a valid weapon detection model.")
            # os._exit(1)
            raise
    