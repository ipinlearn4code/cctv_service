from ultralytics import YOLO
import logging

def load_person_model():
    """Load and return optimized person detection model with fallback"""
    try:
        # Try loading YOLOv11 large model first
        model = YOLO('yolo11l.pt')
        model.fuse()  # Optimize model
        logging.info("Person detection model (yolo11l.pt) loaded and optimized successfully")
        return model
    except Exception as e:
        logging.warning(f"Failed to load yolo11l.pt: {e}")
        try:
            # Fallback to YOLOv8 model
            logging.info("Trying fallback to yolov8n.pt...")
            model = YOLO('yolov8n.pt')
            model.fuse()  # Optimize model
            logging.info("Person detection model (yolov8n.pt) loaded and optimized successfully")
            return model
        except Exception as e2:
            logging.error(f"Failed to load fallback model: {e2}")
            raise