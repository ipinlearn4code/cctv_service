import logging
import time
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import cctv_routes, detection_routes, streaming_routes
from app.core.config import load_detection_config
from app.core.csv_manager import read_csv
from app.services.stream_processor import processor

# Setup logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = load_detection_config()
    logging.info(f"Service started with detection config: {config}")
    
    # Start background processing for all active CCTVs
    try:
        df = read_csv("data/cctv_config.csv", ["id", "name", "ip_address", "location", "status"])
        df["id"] = df["id"].astype(str).str.strip().str.strip('"')
        
        # Start only active cameras
        active_cctvs = df[df["status"] == "active"]
        for _, cctv in active_cctvs.iterrows():
            cctv_id = cctv["id"]
            logging.info(f"Starting background processing for CCTV {cctv_id} at startup")
            processor.start_background_processing(cctv_id)
    except Exception as e:
        logging.error(f"Failed to start background processing: {e}")

    yield  # <-- Here the app runs

    # Shutdown
    logging.info("Service shutting down...")
    processor.stop_background_processing()  # Stop all background processes

# Create FastAPI app
app = FastAPI(title="CCTV Detection Service", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    start_time = time.time()
    
    # Perform basic health checks
    try:
        # Check if processor is available
        processor_status = "healthy" if processor else "unhealthy"
        
        # Check active cameras count
        active_cameras = len(processor.active_cams) if processor and hasattr(processor, 'active_cams') else 0
        
        # Calculate response time
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            "status": "online",
            "response_time_ms": response_time_ms,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "service": "CCTV Detection Service",
            "processor_status": processor_status,
            "active_cameras": active_cameras
        }
    except Exception as e:
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "status": "online",
            "response_time_ms": response_time_ms,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "service": "CCTV Detection Service",
            "error": str(e)
        }

# Include routers
app.include_router(cctv_routes.router, prefix="/cctv", tags=["CCTV"])
app.include_router(detection_routes.router, prefix="/detection_config", tags=["Detection Config"])
app.include_router(streaming_routes.router, prefix="/stream", tags=["Streaming"])
