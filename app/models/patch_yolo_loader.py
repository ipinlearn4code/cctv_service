import torch
import os
import logging
from pathlib import Path
import functools
import sys

def safe_torch_load(file, *args, **kwargs):
    """
    A wrapper for torch.load that handles PyTorch 2.6+ security restrictions
    by forcing weights_only=False for YOLO model loading
    """
    # Override weights_only to False for model loading
    kwargs['weights_only'] = False
    logging.debug(f"Loading model with patched torch.load: {file}")
    return original_torch_load(file, *args, **kwargs)

def patch_ultralytics_loader():
    """
    Patch the Ultralytics/YOLO model loading mechanism to work with PyTorch 2.6+
    This replaces torch.load with our custom version in the ultralytics package.
    """
    try:
        global original_torch_load
        # Store the original torch.load function
        original_torch_load = torch.load
        
        # Replace it with our safe version
        torch.load = safe_torch_load
        
        # Add ultralytics models to safe globals list for good measure
        try:
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
        except:
            logging.warning("Failed to add ultralytics to safe globals - continuing with patched loader")
        
        logging.info("Successfully patched torch.load for YOLO model loading")
        return True
    except Exception as e:
        logging.error(f"Failed to patch torch loading mechanism: {e}")
        return False

def verify_model_paths():
    """Verify the existence of the model files and log their status"""
    model_paths = [
        'data/models/yolo11l.pt',
        'yolov8n.pt',  # Default fallback model
        'data/models/detect/train2/weights/best.pt'
    ]
    
    results = {}
    for model_path in model_paths:
        path = Path(model_path)
        exists = path.exists()
        size = path.stat().st_size if exists else None
        results[model_path] = {
            "exists": exists,
            "size_bytes": size,
            "absolute_path": str(path.absolute()) if exists else None
        }
        
        if exists:
            logging.info(f"Model found: {model_path} ({size / (1024*1024):.2f} MB)")
        else:
            logging.warning(f"Model NOT found: {model_path}, expected at {str(path.absolute())}")
    
    return results

# Apply patch when module is imported
patch_result = patch_ultralytics_loader()
model_status = verify_model_paths()
