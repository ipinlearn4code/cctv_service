import cv2
import os
import time
import logging
from queue import Queue
from threading import Thread
from app.services.file_uploader import upload_file
from app.core.config import load_detection_config

class VideoRecorder:
    def __init__(self, cctv_id: str, duration: int, enable_video: bool, enable_screenshot: bool):
        self.cctv_id = cctv_id
        self.duration = duration
        self.enable_video = enable_video
        self.enable_screenshot = enable_screenshot
        self.frame_queue = Queue()
        self.recording = False
        self.last_record_time = 0

    def add_frame(self, frame):
        if time.time() - self.last_record_time > self.duration:
            self.frame_queue.put(frame)

    def start_recording(self):
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.recording = True
                timestamp = int(time.time())
                files = []

                if self.enable_screenshot:
                    screenshot_path = f"data/temp/{self.cctv_id}_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    files.append(("screenshot", screenshot_path))
                    logging.info(f"Screenshot saved: {screenshot_path}")

                if self.enable_video:
                    video_path = f"data/temp/{self.cctv_id}_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(video_path, fourcc, 30, (frame.shape[1], frame.shape[0]))
                    writer.write(frame)
                    start_time = time.time()
                    while time.time() - start_time < self.duration and not self.frame_queue.empty():
                        frame = self.frame_queue.get()
                        writer.write(frame)
                    writer.release()
                    files.append(("video", video_path))
                    logging.info(f"Video saved: {video_path}")

                # Upload files
                config = load_detection_config()
                for file_type, file_path in files:
                    try:
                        upload_file(file_path, file_type, self.cctv_id, config["external_endpoint"])
                        os.remove(file_path)
                        logging.info(f"File uploaded and deleted: {file_path}")
                    except Exception as e:
                        logging.error(f"Failed to upload {file_path}: {str(e)}")

                self.last_record_time = time.time()
                self.recording = False