import cv2
import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.csv_manager import read_csv, write_csv

def test_camera_connection(cctv_id, ip_address):
    """Test if a camera is reachable"""
    try:
        print(f"Testing CCTV {cctv_id} at {ip_address}...")
        
        cap = cv2.VideoCapture(ip_address)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set timeout
        try:
            cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5 seconds
        except:
            pass
        
        is_connected = False
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                is_connected = True
                print(f"  ✅ CCTV {cctv_id} - Connection successful")
            else:
                print(f"  ❌ CCTV {cctv_id} - Opened but no frame received")
        else:
            print(f"  ❌ CCTV {cctv_id} - Failed to open stream")
        
        cap.release()
        return is_connected
        
    except Exception as e:
        print(f"  ❌ CCTV {cctv_id} - Error: {e}")
        return False

def update_camera_statuses():
    """Test all cameras and update their statuses"""
    try:
        # Read current configuration
        df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
        df["id"] = df["id"].astype(str)
        
        print("Testing all camera connections...")
        print("=" * 60)
        
        updated = False
        
        for index, cctv in df.iterrows():
            cctv_id = str(cctv["id"])
            current_status = str(cctv["status"]).lower()
            ip_address = str(cctv["ip_address"])
            name = str(cctv["name"])
            
            print(f"\nTesting: {name} (ID: {cctv_id})")
            print(f"Current status: {current_status}")
            print(f"URL: {ip_address}")
            
            # Skip disabled cameras
            if current_status == "disabled":
                print(f"  ⏭️  CCTV {cctv_id} - Skipped (disabled)")
                continue
            
            # Test connection
            is_connected = test_camera_connection(cctv_id, ip_address)
            
            # Determine new status
            if is_connected:
                new_status = "active" if current_status in ["disconnect", "disconnected"] else current_status
            else:
                new_status = "disconnect" if current_status == "active" else current_status
            
            # Update status if changed
            if new_status != current_status:
                df.at[index, "status"] = new_status
                updated = True
                print(f"  🔄 Status changed: {current_status} → {new_status}")
            else:
                print(f"  ➡️  Status unchanged: {current_status}")
        
        # Save updates if any
        if updated:
            write_csv("data/cctv_config.csv", df)
            print(f"\n✅ Status updates saved to configuration file")
        else:
            print(f"\n➡️  No status changes needed")
            
        print("\n" + "=" * 60)
        print("Final status summary:")
        status_counts = df["status"].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count} cameras")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    update_camera_statuses()
