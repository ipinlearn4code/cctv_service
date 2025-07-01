import cv2
import numpy as np
import logging
from ultralytics import YOLO
from app.core.csv_manager import read_csv
from app.core.config import load_detection_config
from app.services.recording_manager import VideoRecorder
from app.models.yolo_weapon import load_weapon_model
from app.models.yolo_person import load_person_model
from threading import Thread, Lock
import time
import queue
from collections import defaultdict
import asyncio

class StreamProcessor:
    def __init__(self):
        # Load models using optimized helper functions
        logging.info("Initializing YOLO models...")
        
        try:
            self.model_weapon = load_weapon_model()
        except Exception as e:
            logging.error(f"Failed to load weapon model: {e}")
            raise
        
        try:
            self.model_person = load_person_model()
        except Exception as e:
            logging.error(f"Failed to load person model: {e}")
            raise
        
        logging.info("Models loaded and optimized successfully.")
        
        self.weapon_classes = ['api', 'tajam', 'tumpul']
        self.person_class_index = 0
        self.recorders = {}
        self.config = load_detection_config()
        
        # Cache for CSV data to avoid repeated file reads
        self._csv_cache = {}
        self._csv_cache_time = {}
        self._csv_cache_ttl = 30  # Cache for 30 seconds
        
        # Store the latest processed frames for each CCTV
        self.latest_frames = {}
        self.frame_locks = defaultdict(Lock)  # Locks for thread-safe frame access
        
        # Keep track of active streams
        self.active_cams = {}
        self.cam_locks = defaultdict(Lock)  # Locks for thread-safe camera operations
        
        # Background processing flags
        self.running = True
        self.background_threads = {}
        
        # Performance optimization - track processing times
        self.processing_times = defaultdict(list)
        self.max_processing_times = 100  # Keep track of last N processing times
        
        logging.info("StreamProcessor initialized with optimized weapon and person models.")

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1_b, y1_b, x2_b, y2_b = box2
        
        # Calculate intersection coordinates
        xi1 = max(x1, x1_b)
        yi1 = max(y1, y1_b)
        xi2 = min(x2, x2_b)
        yi2 = min(y2, y2_b)
        
        # Calculate intersection area
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def _are_boxes_close(self, box1, box2, proximity=10):
        """Check if two bounding boxes are within proximity pixels or overlapping."""
        x1, y1, x2, y2 = box1
        x1_b, y1_b, x2_b, y2_b = box2
        
        # Check if boxes overlap (IoU > 0) or are within proximity
        if self._calculate_iou(box1, box2) > 0:
            return True
        
        # Check if boxes are within proximity
        if (x1 - proximity <= x2_b and x2 + proximity >= x1_b and
            y1 - proximity <= y2_b and y2 + proximity >= y1_b):
            return True
        
        return False

    def process_frame(self, frame, cctv_id):
        # Resize image for faster processing if necessary
        height, width = frame.shape[:2]
        if height > 720:  # Resize large frames for faster processing
            scale = 720 / height
            frame_resized = cv2.resize(frame, (int(width * scale), 720))
        else:
            frame_resized = frame
        
        # Run inference with batch size optimization
        result_weapon = self.model_weapon(frame_resized, conf=0.5, verbose=False)[0]
        result_person = self.model_person(frame_resized, conf=0.3, verbose=False)[0]
        weapon_detected = False
        person_crowd_detected = False
        
        # Create a copy of the original frame for drawing
        annotated_frame = frame.copy()

        # Process weapon detections
        for box in result_weapon.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            
            # Get coordinates and adjust if we resized
            coords = box.xyxy[0].tolist()
            if height > 720:
                coords = [c / scale for c in coords]
                
            x1, y1, x2, y2 = map(int, coords)
            label = self.weapon_classes[cls_id] if cls_id < len(self.weapon_classes) else f"ID:{cls_id}"
            weapon_detected = True
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Process person detections
        count_person = 0
        person_boxes = []
        for box in result_person.boxes:
            if int(box.cls) == self.person_class_index:
                count_person += 1
                
                # Get coordinates and adjust if we resized
                coords = box.xyxy[0].tolist()
                if height > 720:
                    coords = [c / scale for c in coords]
                
                person_boxes.append(coords)
                
                x1, y1, x2, y2 = map(int, coords)
                conf = float(box.conf)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'{conf:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check for crowded person detection (15 or more persons with close/overlapping boxes)
        if count_person >= 15:
            proximity = 10  # Proximity threshold in pixels
            groups = []
            visited = set()
            
            for i, box in enumerate(person_boxes):
                if i not in visited:
                    current_group = [i]
                    visited.add(i)
                    for j, other_box in enumerate(person_boxes):
                        if j not in visited and i != j:
                            if self._are_boxes_close(box, other_box, proximity):
                                current_group.append(j)
                                visited.add(j)
                    groups.append(current_group)
            
            # If any group has 15 or more persons, trigger crowd detection
            for group in groups:
                if len(group) >= 15:
                    person_crowd_detected = True
                    break

        # Add counts to frame
        y_offset = 30
        cv2.putText(annotated_frame, f'Person: {count_person}', (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_offset += 30
        
        # Precalculate weapon counts
        weapon_counts = [0] * len(self.weapon_classes)
        for box in result_weapon.boxes:
            cls_id = int(box.cls)
            if cls_id < len(self.weapon_classes):
                weapon_counts[cls_id] += 1
        
        # Add weapon counts
        for i, cls in enumerate(self.weapon_classes):
            cv2.putText(annotated_frame, f'{cls}: {weapon_counts[i]}', (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            y_offset += 30

        # Add crowd detection status
        if person_crowd_detected:
            cv2.putText(annotated_frame, 'Crowd Detected', (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Handle recording - trigger for weapons or crowded persons
        if (weapon_detected or person_crowd_detected) and (self.config["enable_video"] or self.config["enable_screenshot"]):
            self._submit_frame_for_recording(cctv_id, annotated_frame.copy())

        return annotated_frame

    def start_background_processing(self, cctv_id):
        """Start background processing for a specific CCTV camera"""
        with self.cam_locks[cctv_id]:
            if cctv_id in self.background_threads and self.background_threads[cctv_id].is_alive():
                logging.info(f"Background processing already running for CCTV {cctv_id}")
                return
            
            # Get camera information with caching
            df = self._get_cached_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
            df["id"] = df["id"].astype(str).str.strip().str.strip('"')
            cctv_id_clean = str(cctv_id).strip().strip('"')
            
            cctv = df[df["id"] == cctv_id_clean]
            if cctv.empty:
                logging.error(f"Cannot start background processing: CCTV with id '{cctv_id_clean}' not found.")
                return
            
            # Create and start the background thread
            thread = Thread(
                target=self._process_camera_feed,
                args=(cctv_id, cctv.iloc[0]["ip_address"]),
                daemon=True
            )
            thread.start()
            self.background_threads[cctv_id] = thread
            logging.info(f"Started background processing for CCTV {cctv_id}")
    
    def stop_background_processing(self, cctv_id=None):
        """Stop background processing for a specific CCTV or all CCTVs"""
        if cctv_id is None:
            # Stop all background processing
            self.running = False
            logging.info("Stopping all background processing")
            # Wait for threads to finish
            for thread_id, thread in self.background_threads.items():
                thread.join(timeout=2.0)
            self.background_threads.clear()
            self.running = True
        else:
            # Stop specific camera
            with self.cam_locks[cctv_id]:
                if cctv_id in self.active_cams:
                    self.active_cams[cctv_id] = False
                    if cctv_id in self.background_threads:
                        self.background_threads[cctv_id].join(timeout=2.0)
                        del self.background_threads[cctv_id]
                    logging.info(f"Stopped background processing for CCTV {cctv_id}")
    
    def _process_camera_feed(self, cctv_id, ip_address):
        """Background thread function to continuously process a camera feed"""
        logging.info(f"Starting camera feed processing for {cctv_id} at {ip_address}")
        
        # Set OpenCV buffer size to minimum to reduce latency
        cap = cv2.VideoCapture(ip_address)
        if not cap.isOpened():
            logging.warning(f"Failed to open camera feed for CCTV {cctv_id}: {ip_address}")
            return
            
        # Set buffer size to minimum to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        with self.cam_locks[cctv_id]:
            self.active_cams[cctv_id] = True
        
        try:
            failure_count = 0
            frame_skip = 0  # For adaptive frame skipping
            
            while self.active_cams.get(cctv_id, False):
                # Measure start time for processing rate calculation
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    failure_count += 1
                    if failure_count <= 3:
                        logging.debug(f"Failed to read frame from CCTV {cctv_id} (attempt {failure_count})")
                    elif failure_count == 4:
                        logging.warning(f"Multiple frame read failures for CCTV {cctv_id}")
                    
                    if failure_count > 5:
                        logging.warning(f"Too many failures reading from CCTV {cctv_id}, attempting reconnection...")
                        cap.release()
                        time.sleep(1)
                        cap = cv2.VideoCapture(ip_address)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        failure_count = 0
                        if not cap.isOpened():
                            logging.warning(f"Failed to reconnect to camera feed for CCTV {cctv_id}: {ip_address}")
                            break
                    time.sleep(0.1)
                    continue
                
                # Reset failure count on successful frame read
                failure_count = 0
                
                # Implement adaptive frame skipping based on system load
                if frame_skip > 0:
                    frame_skip -= 1
                    continue
                
                try:
                    # Process the frame
                    processed_frame = self.process_frame(frame, cctv_id)
                    
                    # Update the latest frame for this camera (thread-safe)
                    with self.frame_locks[cctv_id]:
                        self.latest_frames[cctv_id] = processed_frame.copy()
                        
                    # Calculate processing time
                    process_time = time.time() - start_time
                    
                    # Track processing times for adaptive adjustment
                    self.processing_times[cctv_id].append(process_time)
                    if len(self.processing_times[cctv_id]) > self.max_processing_times:
                        self.processing_times[cctv_id].pop(0)
                    
                    # Adjust frame skip based on processing time
                    avg_process_time = sum(self.processing_times[cctv_id]) / len(self.processing_times[cctv_id])
                    
                    # If processing is slow (> 50ms per frame), skip frames
                    if avg_process_time > 0.05:
                        frame_skip = max(0, int(avg_process_time / 0.033) - 1)
                    
                    # Log performance metrics periodically
                    if len(self.processing_times[cctv_id]) % 100 == 0:
                        logging.info(f"CCTV {cctv_id} avg processing time: {avg_process_time*1000:.1f}ms, frame skip: {frame_skip}")
                        
                except Exception as e:
                    logging.error(f"Error processing frame for {cctv_id}: {e}")
                
                # Adaptive sleep time - sleep less if processing took longer
                sleep_time = max(0.001, 0.033 - process_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except Exception as e:
            logging.exception(f"Unexpected error in camera processing thread for {cctv_id}: {e}")
        finally:
            cap.release()
            with self.cam_locks[cctv_id]:
                self.active_cams[cctv_id] = False
            logging.info(f"Stopped camera feed processing for {cctv_id}")
    
    def get_latest_frame(self, cctv_id):
        """Get the latest processed frame for a camera (thread-safe)"""
        with self.frame_locks[cctv_id]:
            if cctv_id in self.latest_frames:
                return self.latest_frames[cctv_id].copy()
        return None

    def _submit_frame_for_recording(self, cctv_id, frame):
        """
        Non-blocking method to handle frame recording when weapons or crowd are detected.
        """
        try:
            # Check if we need to create a new recorder
            if cctv_id not in self.recorders:
                self.recorders[cctv_id] = VideoRecorder(
                    cctv_id, self.config["record_duration"], 
                    self.config["enable_video"], 
                    self.config["enable_screenshot"]
                )
                # Start the recording in a separate thread
                Thread(target=self.recorders[cctv_id].start_recording, daemon=True).start()
                logging.info(f"Started new recording for CCTV {cctv_id} due to detection")
                
            # Use a separate thread to add the frame to avoid blocking the inference
            Thread(
                target=self._add_frame_to_recorder,
                args=(cctv_id, frame),
                daemon=True
            ).start()
            
        except Exception as e:
            logging.error(f"Error submitting frame for recording: {e}")
            
    def _add_frame_to_recorder(self, cctv_id, frame):
        """Helper method to add a frame to a recorder in a separate thread"""
        try:
            if cctv_id in self.recorders:
                self.recorders[cctv_id].add_frame(frame)
        except Exception as e:
            logging.error(f"Error adding frame to recorder: {e}")

    def _get_cached_csv(self, file_path, columns):
        """Get CSV data with caching to reduce file I/O operations"""
        current_time = time.time()
        
        # Check if we have valid cached data
        if (file_path in self._csv_cache and 
            file_path in self._csv_cache_time and
            current_time - self._csv_cache_time[file_path] < self._csv_cache_ttl):
            return self._csv_cache[file_path]
        
        # Read fresh data and cache it
        df = read_csv(file_path, columns)
        self._csv_cache[file_path] = df
        self._csv_cache_time[file_path] = current_time
        return df

    def clear_csv_cache(self):
        """Clear CSV cache to force fresh reads"""
        self._csv_cache.clear()
        self._csv_cache_time.clear()
        logging.info("CSV cache cleared")

# Create processor instance
processor = StreamProcessor()

def generate_frames(cctv_id: str):
    """
    Generate MJPEG frames for streaming by pulling from the background-processed frames
    """
    logging.info(f"Starting MJPEG stream for CCTV id: '{cctv_id}'")
    
    # Clean up the CCTV ID
    cctv_id_clean = str(cctv_id).strip().strip('"')
    
    # Ensure background processing is running for this camera
    processor.start_background_processing(cctv_id_clean)
    
    # Wait for the first frame with a shorter timeout
    start_time = time.time()
    while time.time() - start_time < 3:
        if processor.get_latest_frame(cctv_id_clean) is not None:
            break
        time.sleep(0.05)
    
    # Optimization parameters
    last_frame_time = time.time()
    frame_interval = 0.033  # Target frame interval (30 FPS)
    
    try:
        no_frame_count = 0
        last_frame = None
        
        while True:
            # Get current time
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Only get a new frame if enough time has passed
            if elapsed >= frame_interval:
                frame = processor.get_latest_frame(cctv_id_clean)
                
                if frame is not None:
                    # Reset counter and save last good frame
                    no_frame_count = 0
                    last_frame = frame
                    last_frame_time = current_time
                    
                    # Use higher quality for encoding
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        logging.error("Failed to encode frame to JPEG")
                else:
                    no_frame_count += 1
                    if no_frame_count > 30:
                        logging.error(f"No frames available for CCTV {cctv_id_clean} after multiple attempts")
                        break
                    
                    if last_frame is not None:
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                        ret, buffer = cv2.imencode('.jpg', last_frame, encode_param)
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    time.sleep(0.05)
                    continue
            else:
                sleep_time = max(0.001, frame_interval - elapsed)
                time.sleep(sleep_time)
    
    except GeneratorExit:
        logging.info(f"MJPEG stream for CCTV {cctv_id_clean} closed by client")
    except Exception as e:
        logging.error(f"Unhandled exception in generate_frames: {e}")
        raise