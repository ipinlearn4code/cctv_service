# CCTV Detection Service

A real-time CCTV monitoring system with AI-powered weapon detection, person counting, and crowd detection using YOLOv11 models optimized for NVIDIA GPU acceleration.

## 🚀 Features

- **Real-time AI Detection**: Weapon detection (firearms, sharp weapons, blunt weapons)
- **Person Counting**: Accurate person detection and counting
- **Crowd Detection**: Automatic alerts when 15+ people detected in proximity
- **Live Streaming**: MJPEG video streams with detection overlays
- **Auto Recording**: Automatic video/screenshot capture on threat detection
- **GPU Optimization**: NVIDIA Quadro T1000 acceleration
- **Background Processing**: Multi-threaded processing for multiple cameras
- **Dynamic Status Monitoring**: Automatic camera connection validation and status updates
- **System Monitoring**: Terminal output tracking and system status monitoring
- **RESTful API**: Complete CRUD operations for camera management

## 🔄 Dynamic CCTV Status Management

### **Automated Status Validation**
The service continuously monitors camera connections and automatically updates status:

- **Active → Disconnect**: When a camera becomes unreachable
- **Disconnect → Active**: When connection is restored
- **Disabled**: Cameras are skipped from monitoring (manual control only)

### **Background Monitoring**
- **Check Interval**: Every 10 seconds (configurable)
- **Connection Timeout**: 5 seconds per camera test
- **Smart Caching**: Avoids redundant checks within 30 seconds
- **Thread-Safe**: Non-blocking operation with concurrent processing

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **AI Models**: YOLOv11 (Ultralytics)
- **Computer Vision**: OpenCV
- **GPU**: CUDA support for NVIDIA GPUs
- **Data Storage**: CSV files
- **Streaming**: MJPEG over HTTP

## 🏃‍♂️ Quick Start

1. **Install GPU Dependencies**:
   ```bash
   pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Install Other Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install psutil GPUtil  # For system monitoring
   ```

3. **Start the Server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
   ```

4. **Access the API**:
   - **API Server**: http://localhost:8001
   - **Swagger UI**: http://localhost:8001/docs
   - **ReDoc**: http://localhost:8001/redoc

## 📖 API Documentation

### Base URL
```
http://localhost:8001
```

---

## 🎥 CCTV Management (`/cctv`)

### Create CCTV Camera
**POST** `/cctv`

Creates a new CCTV camera configuration.

**Request Body:**
```json
{
  "name": "Front Gate Camera",
  "ip_address": "rtsp://192.168.1.100:554/stream",
  "location": "Front Gate",
  "status": "active"
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Front Gate Camera",
  "ip_address": "rtsp://192.168.1.100:554/stream",
  "location": "Front Gate",
  "status": "active"
}
```

### Get All CCTV Cameras
**GET** `/cctv`

Retrieves all CCTV cameras.

### Get CCTV Camera by ID
**GET** `/cctv/{id}`

Retrieves a specific CCTV camera by ID.

### Update CCTV Camera
**PUT** `/cctv/{id}`

Updates an existing CCTV camera configuration.

### Delete CCTV Camera
**DELETE** `/cctv/{id}`

Deletes a CCTV camera configuration.

---

## ⚙️ Detection Configuration (`/detection_config`)

### Get Detection Configuration
**GET** `/detection_config`

Retrieves the current detection configuration.

### Update Detection Configuration
**PUT** `/detection_config`

Updates the detection configuration settings.

---

## 📹 Streaming & Processing (`/stream`)

### Live Video Stream
**GET** `/stream/{cctv_id}`

Streams live video with real-time AI detection overlays.

**Parameters:**
- `cctv_id` (string): CCTV camera ID

**Response:**
- **Content-Type**: `multipart/x-mixed-replace; boundary=frame`
- **Format**: Live MJPEG stream with detection annotations

**Detection Features:**
- ✅ **Weapon Detection**: Highlights weapons with red bounding boxes
- ✅ **Person Detection**: Green bounding boxes with confidence scores
- ✅ **Person Counting**: Real-time person count overlay
- ✅ **Crowd Detection**: Red "Crowd Detected" alert for 15+ people

### Start Background Processing
**POST** `/stream/processor/{cctv_id}/start`

Starts background AI processing for a specific camera.

### Stop Background Processing
**POST** `/stream/processor/{cctv_id}/stop`

Stops background AI processing for a specific camera.

### Get Processor Status
**GET** `/stream/processor/status`

Gets the status of all background processors.

---

## 🔄 CCTV Monitoring (`/stream/monitor`)

### Get Monitoring Status
**GET** `/stream/monitor/status`

Gets comprehensive CCTV monitoring service status and statistics.

**Response:**
```json
{
  "monitoring_active": true,
  "check_interval_seconds": 10,
  "total_cameras": 4,
  "status_distribution": {
    "active": 2,
    "disconnect": 1,
    "disabled": 1
  },
  "cache_size": 3,
  "last_check": "2025-06-29 15:30:25 UTC"
}
```

### Start/Stop Monitoring
**POST** `/stream/monitor/start`
**POST** `/stream/monitor/stop`

Start or stop the CCTV status monitoring service.

**Response:**
```json
{
  "message": "CCTV monitoring service started"
}
```

### Force Camera Check
**POST** `/stream/monitor/check/{cctv_id}`

Force an immediate connection check for a specific camera.

**Parameters:**
- `cctv_id` (string): CCTV camera ID

**Response:**
```json
{
  "cctv_id": "1001",
  "previous_status": "active",
  "current_status": "disconnect",
  "connected": false,
  "ip_address": "rtsp://192.168.1.100:554/stream",
  "timestamp": "2025-06-29 15:30:25 UTC"
}
```

**Use Cases:**
- Test camera after network changes
- Immediate validation after configuration updates
- Troubleshooting connection issues

### Update Check Interval
**PUT** `/stream/monitor/interval/{seconds}`

Update the monitoring check interval (minimum 5 seconds).

**Parameters:**
- `seconds` (int): New check interval in seconds

**Response:**
```json
{
  "message": "Monitoring interval updated to 30 seconds",
  "new_interval": 30
}
```

### Clear Connection Cache
**POST** `/stream/monitor/cache/clear`

Clear the connection cache to force fresh checks for all cameras.

**Response:**
```json
{
  "message": "Connection cache cleared"
}
```

---

## 🖥️ System Monitoring (`/stream/system`)

### Get Terminal Selection/Output
**GET** `/stream/system/terminal-selection`

Retrieves the latest terminal output and command history.

**Response:**
```json
{
  "terminal_selection": "INFO: Started background processing for CCTV 1001",
  "recent_logs": [
    "2025-06-29 10:30:15 - INFO - StreamProcessor initialized",
    "2025-06-29 10:30:16 - INFO - Started background processing for CCTV 1001",
    "2025-06-29 10:30:17 - INFO - CCTV 1001 avg processing time: 45.2ms"
  ],
  "command_history": [
    "[2025-06-29 10:30:15] Started processing for CCTV 1001",
    "[2025-06-29 10:30:20] Stopped processing for CCTV 1002"
  ],
  "timestamp": "2025-06-29T10:30:25.123456"
}
```

**Features:**
- Latest terminal output/selection
- Recent application logs (last 10 lines)
- Command history with timestamps
- Real-time monitoring capability

### Get System Status
**GET** `/stream/system/status`

Gets comprehensive system status including CPU, memory, GPU, and process information.

**Response:**
```json
{
  "system": {
    "cpu_percent": 25.4,
    "memory": {
      "used": "8GB",
      "total": "16GB",
      "percent": 50.2
    },
    "disk": {
      "used": "45GB",
      "total": "100GB",
      "percent": 45.0
    }
  },
  "gpu": [
    {
      "name": "NVIDIA Quadro T1000",
      "memory_used": "2048MB",
      "memory_total": "4096MB",
      "memory_percent": 50.0,
      "temperature": "65°C",
      "load": "75%"
    }
  ],
  "processes": [
    {
      "pid": 1234,
      "name": "python",
      "cpu_percent": 15.2,
      "memory_percent": 8.5
    }
  ],
  "cctv_status": {
    "active_cameras": ["1001", "1002"],
    "total_active": 2
  },
  "timestamp": "2025-06-29T10:30:25.123456"
}
```

**Monitoring Features:**
- ✅ **CPU Usage**: Real-time CPU percentage
- ✅ **Memory Usage**: RAM usage and availability
- ✅ **GPU Status**: NVIDIA GPU memory, temperature, and load
- ✅ **Disk Usage**: Storage utilization
- ✅ **Process Monitoring**: Active Python/Uvicorn processes
- ✅ **CCTV Status**: Integration with camera processing status

### Execute System Command
**POST** `/stream/system/execute`

Executes allowed system commands and returns output (restricted for security).

**Request Body:**
```json
{
  "command": "nvidia-smi"
}
```

**Response:**
```json
{
  "command": "nvidia-smi",
  "stdout": "GPU 0: NVIDIA Quadro T1000 (UUID: GPU-12345...)",
  "stderr": "",
  "return_code": 0,
  "timestamp": "2025-06-29T10:30:25.123456"
}
```

**Allowed Commands:**
- `nvidia-smi` - GPU status
- `ps aux | grep python` - Python processes
- `df -h` - Disk usage
- `free -h` - Memory usage
- `top -n 1` - System processes
- `ls -la` - Directory listing
- `pwd` - Current directory
- `whoami` - Current user
- `date` - System date/time

**Security Features:**
- ✅ **Command Whitelist**: Only allowed commands can be executed
- ✅ **Timeout Protection**: Commands timeout after 10 seconds
- ✅ **Output Logging**: All commands logged to terminal history
- ✅ **Error Handling**: Safe error responses for failed commands

---

## 🚨 AI Detection Capabilities

### Weapon Detection
- **Model**: Custom trained YOLOv11 (`runs/detect/train2/weights/best.pt`)
- **Classes**: 
  - `api` - Firearms and guns
  - `tajam` - Sharp weapons (knives, blades)
  - `tumpul` - Blunt weapons (clubs, hammers)
- **Confidence Threshold**: 0.5

### Person Detection
- **Model**: YOLOv11 Large (`yolo11l.pt`) with YOLOv8 fallback
- **Confidence Threshold**: 0.3
- **Features**: 
  - Accurate person counting
  - Crowd detection (15+ people in proximity)

### Performance Optimizations
- **GPU Acceleration**: NVIDIA CUDA support
- **Model Fusion**: Conv+BN layer optimization
- **Adaptive Processing**: Dynamic frame skipping
- **Multi-threading**: Background processing with thread-safe operations

---

## 🚀 Usage Examples

### 1. Monitor System Status
```bash
# Get comprehensive system status
curl "http://localhost:8001/stream/system/status"

# Get terminal output
curl "http://localhost:8001/stream/system/terminal-selection"
```

### 2. Execute System Commands
```bash
# Check GPU status
curl -X POST "http://localhost:8001/stream/system/execute" \
  -H "Content-Type: application/json" \
  -d '{"command": "nvidia-smi"}'

# Check disk usage
curl -X POST "http://localhost:8001/stream/system/execute" \
  -H "Content-Type: application/json" \
  -d '{"command": "df -h"}'
```

### 3. Start CCTV Monitoring
```bash
# Add a camera
curl -X POST "http://localhost:8001/cctv" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Main Entrance",
    "ip_address": "rtsp://192.168.1.100:554/stream",
    "location": "Front Door",
    "status": "active"
  }'

# Start processing
curl -X POST "http://localhost:8001/stream/processor/1001/start"

# View live stream
open http://localhost:8001/stream/1001
```

### 4. CCTV Status Monitoring
```bash
# Check monitoring service status
curl "http://localhost:8001/stream/monitor/status"

# Force check a specific camera
curl -X POST "http://localhost:8001/stream/monitor/check/1001"

# Update monitoring interval to 30 seconds
curl -X PUT "http://localhost:8001/stream/monitor/interval/30"

# Clear connection cache for fresh checks
curl -X POST "http://localhost:8001/stream/monitor/cache/clear"
```

### 5. Camera Status Management
```bash
# Update camera status to disabled (stops monitoring)
curl -X PUT "http://localhost:8001/cctv/1001" \
  -H "Content-Type: application/json" \
  -d '{"status": "disabled"}'

# Re-enable camera (resumes monitoring)
curl -X PUT "http://localhost:8001/cctv/1001" \
  -H "Content-Type: application/json" \
  -d '{"status": "active"}'

# Check if camera recovered from disconnect
curl -X POST "http://localhost:8001/stream/monitor/check/1001"
```

---

## 🔧 Error Handling

### HTTP Status Codes
- **200**: Success
- **400**: Bad Request (invalid command)
- **403**: Forbidden (command not allowed)
- **404**: Resource not found
- **408**: Request Timeout (command timeout)
- **422**: Validation error
- **500**: Internal server error

### Error Response Format
```json
{
  "detail": "Descriptive error message"
}
```

---

## 🐛 Troubleshooting

### System Monitoring Issues

1. **GPU Information Not Available**:
   ```bash
   pip install GPUtil
   ```

2. **Permission Denied for System Commands**:
   - Check user permissions
   - Commands are restricted for security

3. **Terminal Output Empty**:
   - Check if `logs/app.log` exists
   - Verify logging configuration

### Performance Monitoring

Use the system status endpoint to monitor:
- GPU memory usage (should stay < 90%)
- CPU usage (should stay < 80% for optimal performance)
- Memory usage (monitor for memory leaks)
- Active camera processing threads

---

## 📈 Performance Metrics

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: Minimum 8GB, recommended 16GB+
- **CPU**: Multi-core processor for concurrent processing
- **Storage**: SSD recommended for video recording

### Monitoring Capabilities
- **Real-time System Status**: CPU, memory, disk, GPU monitoring
- **Process Tracking**: Active Python/Uvicorn processes
- **Terminal History**: Command and output logging
- **CCTV Integration**: Processing status with system metrics

---

**🎯 Your CCTV Detection Service now includes comprehensive system monitoring and terminal tracking capabilities!**
