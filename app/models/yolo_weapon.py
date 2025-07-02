from ultralytics import YOLO
import logging

def load_weapon_model():
    """Load and return optimized weapon detection model"""
    try:
        model = YOLO('data/models/detect/train3/weights/best.pt')
        model.fuse()  # Optimize model
        logging.info("Weapon detection model loaded and optimized successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to load weapon model: {e}")
        
        try:
            # Try using the default YOLO model as fallback
            logging.info("Attempting to load default model as fallback")
            model = YOLO('yolov8n.pt')
            model.fuse()  # Optimize model
            logging.warning("Using default YOLO model instead of weapon-specific model")
            return model
        except Exception as e2:
            logging.error(f"All attempts to load weapon detection model failed: {e2}")
            raise