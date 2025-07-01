import cv2
import os
import time
import logging
from queue import Queue
from threading import Thread
from app.services.file_uploader import upload_file
from app.core.config import load_detection_config

class VideoRecorder:
    def __init__(self, cctv_id: str, duration: int, enable_video: bool, enable_screenshot: bool, detection_type: str = "unknown"):
        self.cctv_id = cctv_id
        self.duration = duration
        self.enable_video = enable_video
        self.enable_screenshot = enable_screenshot
        self.detection_type = detection_type
        self.frame_queue = Queue(maxsize=300)  # Limit queue size to prevent memory issues
        self.recording = False
        self.last_record_time = 0
        self.skip_counter = 0
        self.max_frames_per_second = 15  # Limit recording to 15 fps to reduce strain
        self.last_frame_time = 0

    def add_frame(self, frame, detection_type="unknown"):
        current_time = time.time()
        
        # Update detection type if provided
        if detection_type != "unknown":
            self.detection_type = detection_type
        
        # Check if we are allowed to record (based on time since last recording)
        if current_time - self.last_record_time < self.duration and self.recording:
            return  # Skip if we're in cooldown period
            
        # Frame rate limiting for recording
        if current_time - self.last_frame_time < 1.0/self.max_frames_per_second:
            self.skip_counter += 1
            if self.skip_counter % 100 == 0:
                logging.debug(f"Skipped {self.skip_counter} frames for recording to maintain FPS limit")
            return
            
        self.last_frame_time = current_time
        
        # Handle queue overflow - avoid blocking
        if self.frame_queue.full():
            try:
                # Discard oldest frame
                self.frame_queue.get_nowait()
                if self.skip_counter % 100 == 0:
                    logging.debug(f"Recording queue full for {self.cctv_id}, discarded oldest frame")
            except:
                pass
                
        # Try to add the frame without blocking
        try:
            self.frame_queue.put_nowait(frame)
        except:
            pass  # Skip if queue is full

    def start_recording(self):
        while True:
            try:
                # Use a timeout to avoid blocking forever
                frame = self.frame_queue.get(timeout=1.0)
                self.recording = True
                timestamp = int(time.time())
                self.last_record_time = time.time()
                files = []
                
                try:
                    # Create temp directory if it doesn't exist
                    os.makedirs("data/temp", exist_ok=True)
                    
                    # Handle screenshot - quick and non-blocking
                    if self.enable_screenshot:
                        screenshot_path = f"data/temp/{self.cctv_id}_{timestamp}.jpg"
                        # Use lower JPEG quality (90 instead of default 95) for faster write
                        cv2.imwrite(screenshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        files.append(("screenshot", screenshot_path))
                        logging.info(f"Screenshot saved: {screenshot_path}")

                    # Handle video recording - more resource intensive
                    if self.enable_video:
                        video_path = f"data/temp/{self.cctv_id}_{timestamp}.mp4"
                        frames = [frame]
                        start_time = time.time()

                        while time.time() - start_time < self.duration:
                            try:
                                next_frame = self.frame_queue.get(timeout=0.1)
                                frames.append(next_frame)
                            except:
                                time.sleep(0.01)
                        
                        actual_duration = time.time() - start_time
                        actual_fps = max(1, len(frames) / actual_duration)

                        logging.info(f"Collected {len(frames)} frames in {actual_duration:.2f}s. Using FPS: {actual_fps:.2f}")
                        
                        # Use a more efficient codec if available
                        try:
                            # Try hardware acceleration if available
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 encoding
                            writer = cv2.VideoWriter(video_path, fourcc, actual_fps, (frame.shape[1], frame.shape[0]))
                            # writer = cv2.VideoWriter(video_path, fourcc, 15, (frame.shape[1], frame.shape[0]))
                            if not writer.isOpened():
                                raise Exception("mp4v codec failed")
                        except:
                            # Fall back to default codec
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            writer = cv2.VideoWriter(video_path, fourcc, actual_fps, (frame.shape[1], frame.shape[0]))
                            # writer = cv2.VideoWriter(video_path, fourcc, 15, (frame.shape[1], frame.shape[0]))
                        
                        # Write the first frame
                        writer.write(frame)
                        
                        # Set maximum frames to collect
                        max_frames = 15 * self.duration  # 15fps * duration in seconds
                        frames_collected = 1
                        start_time = time.time()
                        
                        # Collect frames for the duration or until we reach max frames
                        while time.time() - start_time < self.duration and frames_collected < max_frames:
                            try:
                                # Use a short timeout to keep things moving
                                next_frame = self.frame_queue.get(timeout=0.1)
                                writer.write(next_frame)
                                frames_collected += 1
                            except:
                                # No more frames available, that's ok
                                time.sleep(0.01)
                                
                        # Clean up
                        writer.release()
                        files.append(("video", video_path))
                        logging.info(f"Video saved: {video_path} with {frames_collected} frames")
                    
                except Exception as e:
                    logging.error(f"Error during recording for CCTV {self.cctv_id}: {e}")
                    
                # Reset recording status
                self.recording = False
                
                # Upload files
                config = load_detection_config()
                for file_type, file_path in files:
                    try:
                        upload_file(file_path, file_type, self.cctv_id, config["external_endpoint"], self.detection_type)
                        os.remove(file_path)
                        logging.info(f"File uploaded and deleted: {file_path} (detection: {self.detection_type})")
                    except Exception as e:
                        logging.error(f"Failed to upload {file_path}: {str(e)}")

                self.last_record_time = time.time()
                
            except:
                # Queue was empty, just continue waiting
                time.sleep(0.1)