from ultralytics import YOLO

def load_weapon_model():
    return YOLO('runs/detect/train2/weights/best.pt')