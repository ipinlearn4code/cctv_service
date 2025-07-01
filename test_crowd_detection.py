#!/usr/bin/env python3
"""
Test script for crowd detection functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.stream_processor import StreamProcessor
import numpy as np

def test_crowd_detection():
    """Test the crowd detection algorithm with sample data"""
    processor = StreamProcessor()
    
    # Test case 1: No crowd (only 5 people, well separated)
    print("Test 1: No crowd detection (5 people, well separated)")
    person_boxes = [
        [10, 10, 50, 100],   # Person 1
        [100, 10, 140, 100], # Person 2 
        [200, 10, 240, 100], # Person 3
        [300, 10, 340, 100], # Person 4
        [400, 10, 440, 100], # Person 5
    ]
    
    crowd_groups = processor._find_crowd_groups(person_boxes, 15, 10)
    print(f"Found {len(crowd_groups)} crowd groups")
    print()
    
    # Test case 2: Crowd detected (16 people, overlapping)
    print("Test 2: Crowd detection (16 people, overlapping)")
    person_boxes = []
    # Create a 4x4 grid of overlapping person boxes
    for row in range(4):
        for col in range(4):
            x1 = col * 30 + 10
            y1 = row * 40 + 10
            x2 = x1 + 40  # Overlapping boxes
            y2 = y1 + 80
            person_boxes.append([x1, y1, x2, y2])
    
    crowd_groups = processor._find_crowd_groups(person_boxes, 15, 10)
    print(f"Found {len(crowd_groups)} crowd groups")
    if crowd_groups:
        for i, group in enumerate(crowd_groups):
            print(f"  Group {i+1}: {len(group)} people")
    print()
    
    # Test case 3: Multiple small groups (not crowds)
    print("Test 3: Multiple small groups (not reaching crowd threshold)")
    person_boxes = []
    # Create 3 groups of 5 people each
    for group in range(3):
        for person in range(5):
            x1 = group * 200 + person * 35 + 10
            y1 = 10
            x2 = x1 + 40
            y2 = 90
            person_boxes.append([x1, y1, x2, y2])
    
    crowd_groups = processor._find_crowd_groups(person_boxes, 15, 10)
    print(f"Found {len(crowd_groups)} crowd groups")
    print()
    
    # Test case 4: Mixed scenario (one crowd + scattered individuals)
    print("Test 4: Mixed scenario (one crowd + scattered individuals)")
    person_boxes = []
    
    # Create one crowd of 16 people
    for row in range(4):
        for col in range(4):
            x1 = col * 35 + 10
            y1 = row * 45 + 10
            x2 = x1 + 40
            y2 = y1 + 80
            person_boxes.append([x1, y1, x2, y2])
    
    # Add some scattered individuals
    scattered_positions = [(300, 10), (350, 10), (400, 10), (300, 200), (350, 200)]
    for x, y in scattered_positions:
        person_boxes.append([x, y, x + 30, y + 70])
    
    crowd_groups = processor._find_crowd_groups(person_boxes, 15, 10)
    print(f"Total people: {len(person_boxes)}")
    print(f"Found {len(crowd_groups)} crowd groups")
    if crowd_groups:
        total_in_crowds = sum(len(group) for group in crowd_groups)
        print(f"People in crowds: {total_in_crowds}")
        print(f"Scattered individuals: {len(person_boxes) - total_in_crowds}")

if __name__ == "__main__":
    test_crowd_detection()
