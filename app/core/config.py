import pandas as pd
from app.core.csv_manager import read_csv

def load_detection_config():
    df = read_csv("data/detection_config.csv", ["record_duration", "enable_video", "enable_screenshot", "external_endpoint", "crowd_threshold", "overlap_threshold"])
    if df.empty:
        return {
            "record_duration": 10,
            "enable_video": True,
            "enable_screenshot": True,
            "external_endpoint": "https://example.com/upload",
            "crowd_threshold": 15,
            "overlap_threshold": 10
        }
    config = df.iloc[0].to_dict()
    
    # Convert string boolean values to actual booleans
    config["enable_video"] = str(config["enable_video"]).lower() == "true"
    config["enable_screenshot"] = str(config["enable_screenshot"]).lower() == "true"
    
    # Convert numeric values
    config["record_duration"] = int(config["record_duration"])
    config["crowd_threshold"] = int(config["crowd_threshold"])
    config["overlap_threshold"] = float(config["overlap_threshold"])
    
    return config