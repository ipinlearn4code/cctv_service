import cv2
import numpy as np
import logging
from ultralytics import YOLO
from app.core.csv_manager import read_csv
from app.core.config import load_detection_config
from app.services.recording_manager import VideoRecorder
from threading import Thread
import time

class StreamProcessor:
    def __init__(self):
        self.model_weapon = YOLO('runs/detect/train2/weights/best.pt')
        self.model_person = YOLO('yolov8n.pt')
        self.weapon_classes = ['api', 'tajam', 'tumpul']
        self.person_class_index = 0
        self.recorders = {}
        self.config = load_detection_config()
        logging.info("StreamProcessor initialized with weapon model and person model loaded.")    

    def process_frame(self, frame, cctv_id):
        result_weapon = self.model_weapon(frame, conf=0.5)[0]
        result_person = self.model_person(frame, conf=0.3)[0]
        weapon_detected = False

        # Process weapon detections
        for box in result_weapon.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.weapon_classes[cls_id] if cls_id < len(self.weapon_classes) else f"ID:{cls_id}"
            weapon_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Process person detections
        count_person = 0
        for box in result_person.boxes:
            if int(box.cls) == self.person_class_index:
                count_person += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'person {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add counts to frame
        y_offset = 30
        cv2.putText(frame, f'Person: {count_person}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_offset += 30
        for i, cls in enumerate(self.weapon_classes):
            count = sum(1 for box in result_weapon.boxes if int(box.cls) == i)
            cv2.putText(frame, f'{cls}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            y_offset += 30

        # Handle recording
        if weapon_detected and (config["enable_video"] or config["enable_screenshot"]):
            if cctv_id not in self.recorders:
                self.recorders[cctv_id] = VideoRecorder(cctv_id, config["record_duration"], config["enable_video"], config["enable_screenshot"])
                Thread(target=self.recorders[cctv_id].start_recording, daemon=True).start()
            self.recorders[cctv_id].add_frame(frame)

        return frame

processor = StreamProcessor()

def generate_frames(cctv_id: str):
    logging.info(f"Looking for CCTV with id: '{cctv_id}'")
    df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
    
    # Clean up the ID column to remove quotes and convert to string
    df["id"] = df["id"].astype(str).str.strip().str.strip('"')
    
    logging.info(f"Available CCTV IDs: {df['id'].tolist()}")
    logging.info(f"Data types - cctv_id: {type(cctv_id)}, df['id']: {df['id'].dtype}")
    
    # Ensure cctv_id is also a clean string
    cctv_id_clean = str(cctv_id).strip().strip('"')
    
    cctv = df[df["id"] == cctv_id_clean]
    if cctv.empty:
        logging.error(f"CCTV with id '{cctv_id_clean}' not found in configuration.")
        raise ValueError(f"CCTV with id '{cctv_id_clean}' not found")

    cap = cv2.VideoCapture(cctv.iloc[0]["ip_address"])
    if not cap.isOpened():
        logging.error(f"Failed to open stream: {cctv.iloc[0]['ip_address']}")
        raise ValueError(f"Failed to open stream: {cctv.iloc[0]['ip_address']}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.error(f"Failed to read frame from stream: {cctv.iloc[0]['ip_address']}")
                time.sleep(1)  # Wait before retrying
                continue  # Try to read the next frame

            try:
                frame = processor.process_frame(frame, cctv_id)
            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                continue

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                logging.error("Failed to encode frame to JPEG.")
            time.sleep(0.033)  # ~30 FPS

    except GeneratorExit:
        logging.info(f"Stream for CCTV {cctv_id} closed by client.")
    except Exception as e:
        logging.error(f"Unhandled exception in generate_frames: {e}")
    finally:
        cap.release()
        logging.info(f"Released video capture for CCTV {cctv_id}.")