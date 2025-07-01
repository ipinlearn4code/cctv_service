import cv2
import asyncio
import logging
import time
from typing import Dict
from threading import Thread, Lock
from app.core.csv_manager import read_csv, write_csv
import pandas as pd

class CCTVStatusMonitor:
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.status_lock = Lock()
        self.connection_cache: Dict[str, Dict] = {}
        self.check_interval = 10  # Check every 10 seconds
        self.connection_timeout = 5  # 5 seconds timeout for connection attempts
        
        logging.info("CCTV Status Monitor initialized")
    
    def start_monitoring(self):
        """Start the background monitoring service"""
        if self.monitoring:
            logging.info("CCTV monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("CCTV status monitoring started")
    
    def stop_monitoring(self):
        """Stop the background monitoring service"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logging.info("CCTV status monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in background thread"""
        while self.monitoring:
            try:
                self._check_all_cameras()
                time.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Error in CCTV monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_all_cameras(self):
        """Check status of all cameras and update accordingly"""
        try:
            # Read current CCTV configuration
            df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
            df["id"] = df["id"].astype(str)
            
            updated = False
            
            for index, cctv in df.iterrows():
                cctv_id = str(cctv["id"])
                current_status = str(cctv["status"]).lower()
                ip_address = str(cctv["ip_address"])
                
                # Skip disabled cameras
                if current_status == "disabled":
                    continue
                
                # Check connection status
                is_connected = self._test_camera_connection(cctv_id, ip_address)
                new_status = self._determine_new_status(current_status, is_connected)
                
                # Update status if changed
                if new_status != current_status:
                    with self.status_lock:
                        df.at[index, "status"] = new_status
                        updated = True
                        logging.info(f"CCTV {cctv_id} status changed: {current_status} -> {new_status}")
            
            # Write back to CSV if any updates occurred
            if updated:
                write_csv("data/cctv_config.csv", df)
                logging.info("CCTV status updates saved to configuration file")
                
        except Exception as e:
            logging.error(f"Error checking camera statuses: {e}")
    
    def _test_camera_connection(self, cctv_id: str, ip_address: str) -> bool:
        """Test if a camera is reachable and streaming"""
        try:
            # Use cached result if recent enough (within 30 seconds)
            cache_key = f"{cctv_id}_{ip_address}"
            current_time = time.time()
            
            if cache_key in self.connection_cache:
                cache_entry = self.connection_cache[cache_key]
                if current_time - cache_entry["timestamp"] < 30:
                    return cache_entry["connected"]
            
            # Test actual connection
            logging.debug(f"Testing connection for CCTV {cctv_id} at {ip_address}")
            
            # Create a VideoCapture object with timeout
            cap = cv2.VideoCapture(ip_address)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set timeout properties if supported
            try:
                cap.set(cv2.CAP_PROP_TIMEOUT, self.connection_timeout * 1000)  # milliseconds
            except:
                pass  # Some versions might not support this
            
            is_connected = False
            
            if cap.isOpened():
                # Try to read a frame to verify the stream is working
                ret, frame = cap.read()
                if ret and frame is not None:
                    is_connected = True
                    logging.debug(f"CCTV {cctv_id} connection successful")
                else:
                    logging.info(f"CCTV {cctv_id} opened but no frame received")
            else:
                logging.info(f"CCTV {cctv_id} failed to open stream at {ip_address}")
            
            cap.release()
            
            # Cache the result
            self.connection_cache[cache_key] = {
                "connected": is_connected,
                "timestamp": current_time
            }
            
            # Log connection status changes
            if not is_connected:
                logging.warning(f"CCTV {cctv_id} connection failed - will update status to 'disconnect'")
            
            return is_connected
            
        except Exception as e:
            logging.warning(f"Connection test failed for CCTV {cctv_id}: {e}")
            return False
    
    def _determine_new_status(self, current_status: str, is_connected: bool) -> str:
        """Determine the new status based on current status and connection result"""
        current_status = current_status.lower()
        
        if current_status == "disabled":
            return "disabled"  # Never change disabled status
        
        if is_connected:
            if current_status in ["disconnect", "disconnected"]:
                return "active"  # Restore to active if connection recovered
            else:
                return current_status  # Keep current status if already active
        else:
            if current_status == "active":
                return "disconnect"  # Change to disconnect if connection lost
            else:
                return current_status  # Keep disconnect status
    
    def force_check_camera(self, cctv_id: str) -> Dict:
        """Force an immediate check of a specific camera"""
        try:
            df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
            df["id"] = df["id"].astype(str)
            
            cctv = df[df["id"] == str(cctv_id)]
            if cctv.empty:
                return {"error": f"CCTV {cctv_id} not found"}
            
            cctv_data = cctv.iloc[0]
            current_status = str(cctv_data["status"]).lower()
            ip_address = str(cctv_data["ip_address"])
            
            if current_status == "disabled":
                return {
                    "cctv_id": cctv_id,
                    "status": current_status,
                    "message": "Camera is disabled, skipping connection check"
                }
            
            # Clear cache for this camera to force fresh check
            cache_key = f"{cctv_id}_{ip_address}"
            if cache_key in self.connection_cache:
                del self.connection_cache[cache_key]
            
            # Test connection
            is_connected = self._test_camera_connection(cctv_id, ip_address)
            new_status = self._determine_new_status(current_status, is_connected)
            
            # Update status if changed
            if new_status != current_status:
                with self.status_lock:
                    df.loc[df["id"] == str(cctv_id), "status"] = new_status
                    write_csv("data/cctv_config.csv", df)
                    logging.info(f"Forced status update for CCTV {cctv_id}: {current_status} -> {new_status}")
            
            return {
                "cctv_id": cctv_id,
                "previous_status": current_status,
                "current_status": new_status,
                "connected": is_connected,
                "ip_address": ip_address,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
            
        except Exception as e:
            logging.error(f"Error in force check for CCTV {cctv_id}: {e}")
            return {"error": str(e)}
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status and statistics"""
        try:
            df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
            
            status_counts = df["status"].value_counts().to_dict()
            
            return {
                "monitoring_active": self.monitoring,
                "check_interval_seconds": self.check_interval,
                "total_cameras": len(df),
                "status_distribution": status_counts,
                "cache_size": len(self.connection_cache),
                "last_check": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
        except Exception as e:
            return {"error": str(e)}
    
    def update_check_interval(self, interval_seconds: int):
        """Update the monitoring check interval"""
        if interval_seconds < 5:
            raise ValueError("Check interval must be at least 5 seconds")
        
        self.check_interval = interval_seconds
        logging.info(f"CCTV monitoring interval updated to {interval_seconds} seconds")
    
    def clear_connection_cache(self):
        """Clear the connection cache to force fresh checks"""
        with self.status_lock:
            self.connection_cache.clear()
            logging.info("CCTV connection cache cleared")

# Create global monitor instance
cctv_monitor = CCTVStatusMonitor()
