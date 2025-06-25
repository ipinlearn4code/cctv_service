import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import cctv_routes, detection_routes, streaming_routes
from app.core.config import load_detection_config

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

    yield  # <-- Here the app runs

    # Shutdown (opsional)
    logging.info("Service shutting down...")

# Create FastAPI app
app = FastAPI(title="CCTV Detection Service", lifespan=lifespan)

# Include routers
app.include_router(cctv_routes.router, prefix="/cctv", tags=["CCTV"])
app.include_router(detection_routes.router, prefix="/detection_config", tags=["Detection Config"])
app.include_router(streaming_routes.router, prefix="/stream", tags=["Streaming"])
