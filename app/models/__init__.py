"""
YOLO Model Loading Functions

This module provides optimized YOLO model loading functions with 
automatic fallback support and performance optimization.
"""

from .yolo_weapon import load_weapon_model
from .yolo_person import load_person_model

__all__ = ['load_weapon_model', 'load_person_model']

