import pandas as pd
from app.core.csv_manager import read_csv

def load_detection_config():
    df = read_csv("data/detection_config.csv", ["record_duration", "enable_video", "enable_screenshot", "external_endpoint"])
    if df.empty:
        return {
            "record_duration": 10,
            "enable_video": True,
            "enable_screenshot": True,
            "external_endpoint": "https://example.com/upload"
        }
    return df.iloc[0].to_dict()