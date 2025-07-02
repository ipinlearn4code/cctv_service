"""
YOLO Model Loading Functions

This module provides optimized YOLO model loading functions with 
automatic fallback support and performance optimization.
"""

# First, apply the patch to handle PyTorch 2.6+ security changes
from .patch_yolo_loader import patch_ultralytics_loader, verify_model_paths

# Then import the model loaders
from .yolo_weapon import load_weapon_model
from .yolo_person import load_person_model

__all__ = ['load_weapon_model', 'load_person_model']

