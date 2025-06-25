from ultralytics import YOLO

def load_person_model():
    return YOLO('yolov8n.pt')