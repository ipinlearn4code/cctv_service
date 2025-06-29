from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, List
import subprocess
import logging
import os
import psutil
from app.services.stream_processor import generate_frames, processor
from app.core.csv_manager import read_csv

router = APIRouter()

# Store terminal outputs and commands for monitoring
terminal_history = []
MAX_HISTORY_SIZE = 100

@router.get("/{cctv_id}")
async def stream(cctv_id: str):
    df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
    if cctv_id not in df["id"].astype(str).values:
        raise HTTPException(status_code=404, detail=f"CCTV id '{cctv_id}' not found")
    try:
        return StreamingResponse(
            generate_frames(cctv_id),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Streaming error: " + str(e))

@router.post("/processor/{cctv_id}/start")
async def start_processing(cctv_id: str):
    """Start background processing for a specific CCTV camera"""
    try:
        # Check if CCTV exists
        df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
        df["id"] = df["id"].astype(str).str.strip().str.strip('"')
        cctv_id_clean = str(cctv_id).strip().strip('"')
        
        if cctv_id_clean not in df["id"].values:
            raise HTTPException(status_code=404, detail=f"CCTV with id '{cctv_id_clean}' not found")
        
        # Start background processing
        processor.start_background_processing(cctv_id_clean)
        
        # Log the command
        _add_to_terminal_history(f"Started processing for CCTV {cctv_id_clean}")
        
        return {"message": f"Started background processing for CCTV {cctv_id_clean}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/processor/{cctv_id}/stop")
async def stop_processing(cctv_id: str):
    """Stop background processing for a specific CCTV camera"""
    try:
        cctv_id_clean = str(cctv_id).strip().strip('"')
        processor.stop_background_processing(cctv_id_clean)
        
        # Log the command
        _add_to_terminal_history(f"Stopped processing for CCTV {cctv_id_clean}")
        
        return {"message": f"Stopped background processing for CCTV {cctv_id_clean}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processor/status")
async def get_processor_status():
    """Get status of all background processors"""
    try:
        active_cams = list(processor.active_cams.keys())
        return {
            "active_cameras": active_cams,
            "total_active": len(active_cams)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/terminal-selection")
async def get_terminal_selection():
    """Get the latest terminal output or selection"""
    try:
        # Get system logs from the application log file
        log_file_path = "logs/app.log"
        terminal_output = []
        
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Get last 10 lines
                    terminal_output = [line.strip() for line in lines[-10:] if line.strip()]
            except Exception as e:
                logging.warning(f"Could not read log file: {e}")
        
        # Get recent terminal history
        recent_history = terminal_history[-5:] if terminal_history else []
        
        return {
            "terminal_selection": terminal_output[-1] if terminal_output else "No recent output",
            "recent_logs": terminal_output,
            "command_history": recent_history,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/status")
async def get_system_status():
    """Get comprehensive system status including CPU, memory, and GPU info"""
    try:
        # Get CPU and memory info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get GPU info if available
        gpu_info = []
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    "name": gpu.name,
                    "memory_used": f"{gpu.memoryUsed}MB",
                    "memory_total": f"{gpu.memoryTotal}MB",
                    "memory_percent": round((gpu.memoryUsed / gpu.memoryTotal) * 100, 1),
                    "temperature": f"{gpu.temperature}°C",
                    "load": f"{gpu.load * 100}%"
                })
        except ImportError:
            gpu_info = [{"error": "GPUtil not installed"}]
        except Exception as e:
            gpu_info = [{"error": f"GPU info unavailable: {str(e)}"}]
        
        # Get active processes related to our application
        active_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower() or 'uvicorn' in proc.info['name'].lower():
                    active_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return {
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "used": f"{memory.used // (1024**3)}GB",
                    "total": f"{memory.total // (1024**3)}GB",
                    "percent": memory.percent
                },
                "disk": {
                    "used": f"{disk.used // (1024**3)}GB",
                    "total": f"{disk.total // (1024**3)}GB",
                    "percent": round((disk.used / disk.total) * 100, 1)
                }
            },
            "gpu": gpu_info,
            "processes": active_processes[:5],  # Top 5 processes
            "cctv_status": {
                "active_cameras": list(processor.active_cams.keys()),
                "total_active": len(processor.active_cams)
            },
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system/execute")
async def execute_system_command(command: dict):
    """Execute a system command and return output (use with caution)"""
    try:
        cmd = command.get("command", "").strip()
        if not cmd:
            raise HTTPException(status_code=400, detail="Command is required")
        
        # Security: Only allow specific safe commands
        allowed_commands = [
            "nvidia-smi", "ps aux | grep python", "df -h", "free -h", 
            "top -n 1", "ls -la", "pwd", "whoami", "date"
        ]
        
        if not any(cmd.startswith(allowed_cmd) for allowed_cmd in allowed_commands):
            raise HTTPException(status_code=403, detail="Command not allowed for security reasons")
        
        # Execute command
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        
        output = {
            "command": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        # Add to terminal history
        _add_to_terminal_history(f"Executed: {cmd}")
        _add_to_terminal_history(f"Output: {result.stdout[:100]}...")
        
        return output
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Command timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _add_to_terminal_history(entry: str):
    """Add entry to terminal history with timestamp"""
    global terminal_history
    timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    terminal_history.append(f"[{timestamp}] {entry}")
    
    # Keep only last MAX_HISTORY_SIZE entries
    if len(terminal_history) > MAX_HISTORY_SIZE:
        terminal_history = terminal_history[-MAX_HISTORY_SIZE:]
