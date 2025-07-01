from ultralytics import YOLO
import logging

def load_weapon_model():
    """Load and return optimized weapon detection model"""
    try:
        model = YOLO('data/models/detect/train2/weights/best.pt')
        model.fuse()  # Optimize model
        logging.info("Weapon detection model loaded and optimized successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to load weapon model: {e}")
        raise