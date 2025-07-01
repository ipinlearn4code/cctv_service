#!/usr/bin/env python3
"""
Simple test for crowd detection logic without model loading
"""

class MockStreamProcessor:
    """Mock class to test just the crowd detection logic"""
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1_b, y1_b, x2_b, y2_b = box2
        
        # Calculate intersection coordinates
        xi1 = max(x1, x1_b)
        yi1 = max(y1, y1_b)
        xi2 = min(x2, x2_b)
        yi2 = min(y2, y2_b)
        
        # Calculate intersection area
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def _calculate_pixel_distance(self, box1, box2):
        """Calculate minimum pixel distance between two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1_b, y1_b, x2_b, y2_b = box2
        
        # If boxes overlap, distance is 0
        if not (x2 < x1_b or x2_b < x1 or y2 < y1_b or y2_b < y1):
            return 0
        
        # Calculate minimum distance between box edges
        dx = max(0, max(x1_b - x2, x1 - x2_b))
        dy = max(0, max(y1_b - y2, y1 - y2_b))
        return (dx ** 2 + dy ** 2) ** 0.5

    def _are_boxes_connected(self, box1, box2, overlap_threshold):
        """Check if two bounding boxes are connected based on overlap threshold."""
        # Check IoU overlap
        iou = self._calculate_iou(box1, box2)
        if iou > 0.2:  # IoU threshold for overlap
            return True
        
        # Check pixel distance
        pixel_distance = self._calculate_pixel_distance(box1, box2)
        return pixel_distance <= overlap_threshold

    def _find_crowd_groups(self, person_boxes, crowd_threshold, overlap_threshold):
        """Find groups of connected person boxes that form crowds."""
        if len(person_boxes) < crowd_threshold:
            return []
        
        # Create adjacency list for connected boxes
        n = len(person_boxes)
        adjacency = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._are_boxes_connected(person_boxes[i], person_boxes[j], overlap_threshold):
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        
        # Find connected components using DFS
        visited = [False] * n
        crowd_groups = []
        
        def dfs(node, group):
            visited[node] = True
            group.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    dfs(neighbor, group)
        
        for i in range(n):
            if not visited[i]:
                group = []
                dfs(i, group)
                if len(group) >= crowd_threshold:
                    crowd_groups.append(group)
        
        return crowd_groups

def test_crowd_detection():
    """Test the crowd detection algorithm with sample data"""
    processor = MockStreamProcessor()
    
    print("🎯 Crowd Detection Algorithm Test")
    print("=" * 50)
    
    # Test case 1: No crowd (only 5 people, well separated)
    print("\n📍 Test 1: No crowd detection (5 people, well separated)")
    person_boxes = [
        [10, 10, 50, 100],   # Person 1
        [100, 10, 140, 100], # Person 2 
        [200, 10, 240, 100], # Person 3
        [300, 10, 340, 100], # Person 4
        [400, 10, 440, 100], # Person 5
    ]
    
    crowd_groups = processor._find_crowd_groups(person_boxes, 15, 10)
    print(f"   Result: {len(crowd_groups)} crowd groups detected ✅")
    
    # Test case 2: Crowd detected (16 people, overlapping)
    print("\n📍 Test 2: Crowd detection (16 people, overlapping)")
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
    print(f"   Result: {len(crowd_groups)} crowd groups detected")
    if crowd_groups:
        for i, group in enumerate(crowd_groups):
            print(f"   Group {i+1}: {len(group)} people 🚨")
    
    # Test case 3: Multiple small groups (not crowds)
    print("\n📍 Test 3: Multiple small groups (not reaching crowd threshold)")
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
    print(f"   Result: {len(crowd_groups)} crowd groups detected ✅")
    
    # Test case 4: Mixed scenario (one crowd + scattered individuals)
    print("\n📍 Test 4: Mixed scenario (one crowd + scattered individuals)")
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
    print(f"   Total people: {len(person_boxes)}")
    print(f"   Crowd groups: {len(crowd_groups)}")
    if crowd_groups:
        total_in_crowds = sum(len(group) for group in crowd_groups)
        print(f"   People in crowds: {total_in_crowds} 🚨")
        print(f"   Scattered individuals: {len(person_boxes) - total_in_crowds} ✅")
    
    print("\n🎉 Crowd Detection Test Complete!")

if __name__ == "__main__":
    test_crowd_detection()
