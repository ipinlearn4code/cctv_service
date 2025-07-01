## Upload Endpoint

### Endpoint
`POST /api/cctv/upload`

### Access
Public (no authentication required)

### Request Format
`multipart/form-data`

### Required Parameters
- `file`: The media file (image/video) to upload
- `cctv_name`: String representing the camera name
- `detection_type`: String (e.g., crowd, weapon)

### Optional Parameters
- `timestamp`: ISO date string (if not provided, server time is used)

### File Constraints
- Maximum file size: 200MB
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.mp4`, `.avi`, `.mov`

### Storage Format
Files are stored in MinIO using the structure:
```
{camera-name}/{yyyy}/{mm}/{dd}/{detection_type}/{original_filename}.{ext}
```

### Example Upload
```bash
curl -X POST http://your-domain.com/api/cctv/upload \
  -F "file=@detection_image.jpg" \
  -F "cctv_name=camera001" \
  -F "detection_type=person" \
  -F "timestamp=2025-07-01T14:30:00Z"
```

### Response Example (Success)
```json
{
  "success": true,
  "message": "File uploaded successfully",
  "data": {
    "storage_path": "camera001/2025/07/01/person/detection_image.jpg",
    "cctv_name": "camera001",
    "detection_type": "croed",
    "timestamp": "2025-07-01T14:30:00.000000Z",
    "file_size": "1.2 MB",
    "original_filename": "detection_image.jpg"
  }
}
```

### Response Example (Error)
```json
{
  "success": false,
  "message": "Validation failed",
  "errors": {
    "file": ["The file field is required."],
    "cctv_name": ["The cctv name field is required."]
  }
}
```
