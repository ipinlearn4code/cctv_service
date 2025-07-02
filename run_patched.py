#!/usr/bin/env python3
import os
import sys
import torch
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment for YOLO model loading with PyTorch 2.6+"""
    try:
        # Store the original torch.load function
        original_torch_load = torch.load
        
        # Define our wrapper function
        def patched_torch_load(file, *args, **kwargs):
            logger.info(f"Using patched torch.load for file: {file}")
            # Force weights_only=False for all model loading
            kwargs['weights_only'] = False
            return original_torch_load(file, *args, **kwargs)
        
        # Replace torch.load with our patched version
        torch.load = patched_torch_load
        
        # Also add safe globals
        try:
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            logger.info("Added ultralytics models to PyTorch safe globals list")
        except Exception as e:
            logger.warning(f"Failed to add safe globals, but continuing with patched loader: {e}")
        
        logger.info("PyTorch loading patched successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to patch PyTorch: {e}")
        return False

if __name__ == "__main__":
    logger.info("Setting up patched environment for YOLO loading...")
    setup_result = setup_environment()
    
    # Add the project root directory to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to Python path")
    
    # Now import and run the app
    logger.info("Starting the FastAPI application...")
    
    try:
        import uvicorn
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
