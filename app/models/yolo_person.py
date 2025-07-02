from ultralytics import YOLO
import logging

def load_person_model():
    """Load and return optimized person detection model with fallback"""
    try:
        # Try loading YOLOv11 large model first
        model = YOLO('data/models/yolo11l.pt')
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
            
            # Try one more time with a downloaded model
            try:
                logging.info("Attempting to download YOLO model as last resort")
                model = YOLO('yolov8n.pt', download=True)
                model.fuse()
                logging.info("Downloaded person detection model as last resort")
                return model
            except Exception as e3:
                logging.error(f"All attempts to load person detection model failed: {e3}")
                raise