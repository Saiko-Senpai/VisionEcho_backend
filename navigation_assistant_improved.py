import cv2
import torch
import numpy as np
import pyttsx3
import time
from collections import deque
import threading

class AutoNavigationAssistant:
    def __init__(self):

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Initialize text-to-speech engine with optimized settings
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)  # Slightly faster for continuous feedback
        self.engine.setProperty('volume', 0.9)
        
        # Load models
        print("Loading AI models...")
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        self.yolo_model.conf = 0.4
        
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        
        # Comprehensive navigation-relevant object classes (COCO dataset)
        self.navigation_objects = {
            # People and animals
            0: 'person', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            
            # Vehicles and transportation
            1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
            6: 'train', 7: 'truck', 8: 'boat',
            
            # Traffic and road objects
            9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 
            12: 'parking meter', 13: 'bench',
            
            # Sports and outdoor equipment
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            
            # Furniture and household items
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
            60: 'dining table', 61: 'toilet', 62: 'tv', 
            
            # Kitchen and appliances
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
            
            # Electronics and office items
            63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
            67: 'cell phone', 68: 'microwave', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush',
            
            # Food and dining items  
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 
            43: 'knife', 44: 'spoon', 45: 'bowl',
            
            # Clothing and accessories
            26: 'backpack', 27: 'umbrella', 28: 'handbag', 29: 'tie',
            30: 'suitcase',
            
            # Food items (potential obstacles when on floor/surfaces)
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake'
        }
        
        # Priority objects that need immediate attention (safety-critical)
        self.priority_objects = {
            # Moving/dangerous objects
            0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 
            1: 'bicycle', 6: 'train', 16: 'bird', 17: 'cat', 18: 'dog',
            
            # Traffic and safety objects
            9: 'traffic light', 11: 'stop sign', 10: 'fire hydrant'
        }
        
        # Large furniture objects that are major navigation obstacles
        self.furniture_objects = {
            56: 'chair', 57: 'couch', 59: 'bed', 60: 'dining table', 
            62: 'tv', 69: 'oven', 72: 'refrigerator'
        }
        
        # Audio feedback management for continuous operation
        self.last_announcement_time = 0
        self.last_urgent_announcement = 0
        self.is_speaking = False
        
        # Timing intervals for different types of announcements
        self.urgent_interval = 1.0      # 1 second for urgent warnings
        self.important_interval = 3.0   # 3 seconds for important obstacles
        self.general_interval = 5.0     # 5 seconds for general information
        self.clear_path_interval = 8.0  # 8 seconds for "path clear" confirmations
        
        # Detection history for stability and change detection
        self.detection_history = deque(maxlen=5)
        self.last_detections = {}
        
        # Step conversion parameters (adjust based on average human stride)
        self.meters_per_step = 0.6  # Average human step is about 60cm
        self.depth_scale_factor = 4.0  # Adjusted for step-based distance
        self.close_threshold = 3.0      # 5 steps or closer
        self.very_close_threshold = 1.5 # 2-3 steps or closer
        
        # Motion detection for automatic scanning
        self.last_frame = None
        self.motion_threshold = 1000  # Threshold for detecting motion
        self.stationary_time = 0
        self.last_motion_time = time.time()
        
        # Scanning control
        self.last_scan_time = 0
        self.scan_interval = 2.0  # Base scan interval when moving
        self.stationary_scan_interval = 6.0  # Longer interval when stationary
        
    def speak_async(self, text, priority="normal"):
        """Non-blocking speech synthesis with priority handling"""
        if priority == "urgent" or not self.is_speaking:
            if priority == "urgent" and self.is_speaking:
                # Stop current speech for urgent messages
                self.engine.stop()
                time.sleep(0.1)
            
            self.is_speaking = True
            thread = threading.Thread(target=self._speak, args=(text,))
            thread.daemon = True
            thread.start()
    
    def _speak(self, text):
        """Internal speech method"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        finally:
            self.is_speaking = False
    
    def detect_motion(self, current_frame):
        """Detect if user is in motion using frame difference"""
        if self.last_frame is None:
            self.last_frame = current_frame.copy()
            return True
        
        # Convert to grayscale for motion detection
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray_current, gray_last)
        motion_pixels = np.sum(diff > 30)  # Count significantly changed pixels
        
        self.last_frame = current_frame.copy()
        
        current_time = time.time()
        if motion_pixels > self.motion_threshold:
            self.last_motion_time = current_time
            self.stationary_time = 0
            return True
        else:
            self.stationary_time = current_time - self.last_motion_time
            return False
    
    def get_position_description(self, x_center, width):
        """Get position descriptions optimized for navigation"""
        relative_pos = x_center / width
        if relative_pos < 0.15:
            return "far left"
        elif relative_pos < 0.35:
            return "left"
        elif relative_pos < 0.65:
            return "directly ahead"
        elif relative_pos < 0.85:
            return "right"
        else:
            return "far right"
    
    def meters_to_steps(self, distance_meters):
        """Convert distance in meters to steps"""
        steps = max(1, round(distance_meters / self.meters_per_step))
        return steps
    
    def estimate_distance_in_steps(self, depth_value, box_area, frame_area):
        """Estimate distance in steps using depth and object size"""
        # Normalize depth (closer objects have higher values in MiDaS output)
        normalized_depth = 1.0 - depth_value
        
        # Scale to real-world distance
        distance_meters = normalized_depth * self.depth_scale_factor
        
        # Adjust based on object size
        size_factor = (box_area / frame_area) * 1.5
        adjusted_distance = max(0.3, distance_meters - size_factor)
        
        # Convert to steps
        steps = self.meters_to_steps(adjusted_distance)
        return steps
    
    def classify_urgency(self, steps, object_class):
        """Classify detection urgency based on steps and object type"""
        is_priority = object_class in self.priority_objects
        is_furniture = object_class in self.furniture_objects
        
        if steps <= 2:
            if is_priority:
                return "urgent"
            elif is_furniture:
                return "warning"
            else:
                return "caution"
        elif steps <= 5:
            if is_priority:
                return "caution"
            elif is_furniture:
                return "info"
            else:
                return "info"
        else:
            return "info"
    
    def filter_detections(self, detections, frame_shape):
        """Filter and prioritize detections"""
        height, width = frame_shape[:2]
        frame_area = height * width
        filtered = []
        
        for det in detections:
            confidence = det[4]
            class_id = int(det[5])
            
            if class_id in self.navigation_objects and confidence > 0.35:
                x1, y1, x2, y2 = map(int, det[:4])
                box_area = (x2 - x1) * (y2 - y1)
                
                if box_area > frame_area * 0.002:  # Filter very small objects
                    filtered.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self.navigation_objects[class_id],
                        'box_area': box_area
                    })
        
        return filtered
    
    def generate_navigation_guidance(self, frame, detections, depth_map):
        """Generate navigation guidance with step-based distances"""
        height, width = frame.shape[:2]
        frame_area = height * width
        
        # Normalize depth map
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        filtered_detections = self.filter_detections(detections, frame.shape)
        
        urgent_warnings = []
        important_obstacles = []
        general_info = []
        
        for det in filtered_detections:
            x1, y1, x2, y2 = det['box']
            x_center = (x1 + x2) / 2
            
            # Get depth for object region
            object_depth = depth_normalized[y1:y2, x1:x2]
            if object_depth.size > 0:
                avg_depth = np.mean(object_depth)
                steps = self.estimate_distance_in_steps(avg_depth, det['box_area'], frame_area)
                
                position = self.get_position_description(x_center, width)
                urgency = self.classify_urgency(steps, det['class_id'])
                
                # Format step description
                if steps == 1:
                    step_str = "1 step"
                else:
                    step_str = f"{steps} steps"
                
                description = f"{det['class_name']} {position}, {step_str} away"
                
                if urgency == "urgent":
                    urgent_warnings.append(f"Caution! {description}")
                elif urgency == "warning" or urgency == "caution":
                    important_obstacles.append(description)
                else:
                    general_info.append(description)
        
        # Check for path obstacles and walls/dead ends
        path_obstacles = self.detect_path_obstacles(depth_normalized, width)
        wall_status = self.detect_wall_or_deadend(depth_normalized, width)
        
        # Add wall/dead end detection to important obstacles
        if wall_status:
            important_obstacles.append(wall_status)
        elif path_obstacles:
            important_obstacles.extend(path_obstacles)
        
        return urgent_warnings, important_obstacles, general_info
    
    def detect_path_obstacles(self, depth_map, width):
        """Detect obstacles in walking path including walls and dead ends"""
        height = depth_map.shape[0]
        obstacles = []
        
        # Focus on center path, lower portion of frame
        path_region = depth_map[height//2:, width//3:2*width//3]
        
        if path_region.size == 0:
            return obstacles
        
        # Check for general obstacles in path
        close_pixels = np.sum(path_region > 0.75)
        total_pixels = path_region.size
        
        if close_pixels / total_pixels > 0.2:
            # Estimate steps for path obstacle
            avg_depth = np.mean(path_region[path_region > 0.75])
            steps = self.meters_to_steps((1.0 - avg_depth) * self.depth_scale_factor)
            step_str = "1 step" if steps == 1 else f"{steps} steps"
            obstacles.append(f"obstacle in path, {step_str} ahead")
        
        return obstacles
    
    def detect_wall_or_deadend(self, depth_map, width):
        """Detect walls and dead ends using comprehensive depth analysis"""
        height = depth_map.shape[0]
        
        # Analyze different regions of the frame
        # Front center region (main path ahead)
        front_center = depth_map[height//3:, width//3:2*width//3]
        
        # Left and right regions for alternative paths
        left_region = depth_map[height//3:, :width//3]
        right_region = depth_map[height//3:, 2*width//3:]
        
        # Upper region (distant view)
        upper_region = depth_map[:height//3, width//4:3*width//4]
        
        wall_detected = False
        deadend_detected = False
        alternative_paths = []
        
        # Check for wall directly ahead (high depth values indicate close surfaces)
        if front_center.size > 0:
            # Look for consistent high depth values (close surfaces) across the front
            high_depth_pixels = np.sum(front_center > 0.8)
            total_front_pixels = front_center.size
            
            # If more than 40% of the front view shows very close surfaces
            if high_depth_pixels / total_front_pixels > 0.4:
                # Check if it's a consistent wall (not just scattered objects)
                # Analyze horizontal consistency in the upper part of front region
                upper_front = front_center[:front_center.shape[0]//2, :]
                if upper_front.size > 0:
                    # Check each row for consistency
                    consistent_rows = 0
                    for row in upper_front:
                        if np.sum(row > 0.8) / len(row) > 0.3:
                            consistent_rows += 1
                    
                    # If most rows show consistent high depth, it's likely a wall
                    if consistent_rows / upper_front.shape[0] > 0.6:
                        wall_detected = True
                        
                        # Estimate distance to wall
                        avg_wall_depth = np.mean(front_center[front_center > 0.8])
                        wall_distance_steps = self.meters_to_steps((1.0 - avg_wall_depth) * self.depth_scale_factor)
                        
        # Check for alternative paths (left and right)
        if wall_detected:
            # Check left path
            if left_region.size > 0:
                left_clear_pixels = np.sum(left_region < 0.6)  # Lower depth = more distant/clear
                if left_clear_pixels / left_region.size > 0.3:
                    alternative_paths.append("left")
            
            # Check right path  
            if right_region.size > 0:
                right_clear_pixels = np.sum(right_region < 0.6)
                if right_clear_pixels / right_region.size > 0.3:
                    alternative_paths.append("right")
            
            # If no alternative paths found, it's a dead end
            if not alternative_paths:
                deadend_detected = True
        
        # Return appropriate message
        if deadend_detected:
            return "dead end reached. Turn around"
        elif wall_detected:
            if alternative_paths:
                if len(alternative_paths) == 2:
                    return f"wall ahead, {wall_distance_steps} steps. Turn left or right"
                else:
                    direction = alternative_paths[0]
                    return f"wall ahead, {wall_distance_steps} steps. Turn {direction}"
            else:
                return f"wall ahead, {wall_distance_steps} steps"
        
        return None
    
    def should_announce(self, urgent_warnings, important_obstacles, general_info, is_moving):
        """Determine if announcement should be made based on priority and timing"""
        current_time = time.time()
        
        # Check for wall/dead end messages (highest priority after urgent)
        wall_messages = [msg for msg in important_obstacles if any(keyword in msg.lower() 
                        for keyword in ['wall', 'dead end', 'turn around'])]
        
        # Always announce urgent warnings with minimal delay
        if urgent_warnings:
            if current_time - self.last_urgent_announcement > self.urgent_interval:
                self.last_urgent_announcement = current_time
                return "urgent", urgent_warnings[:1]
        
        # Wall/dead end messages get high priority
        if wall_messages:
            if current_time - self.last_announcement_time > self.urgent_interval * 1.5:
                return "wall", wall_messages[:1]
        
        # Important obstacles - announce more frequently when moving
        if important_obstacles and not wall_messages:
            interval = self.important_interval if is_moving else self.important_interval * 1.5
            if current_time - self.last_announcement_time > interval:
                return "important", important_obstacles[:2]
        
        # General information - less frequent
        if general_info:
            interval = self.general_interval if is_moving else self.general_interval * 2
            if current_time - self.last_announcement_time > interval:
                return "general", general_info[:2]
        
        # Only announce clear path if no walls detected and no other obstacles
        if not urgent_warnings and not important_obstacles and not general_info:
            if current_time - self.last_announcement_time > self.clear_path_interval:
                return "clear", ["Path appears clear"]
        
        return None, []
    
    def should_scan(self, is_moving):
        """Determine if a new scan should be performed"""
        current_time = time.time()
        
        if is_moving:
            return current_time - self.last_scan_time > self.scan_interval
        else:
            # Less frequent scanning when stationary, but still scan occasionally
            return current_time - self.last_scan_time > self.stationary_scan_interval
    
    def run(self):
        """Main automatic navigation assistance loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)
        
        print("🚶 Automatic Navigation Assistant Starting...")
        print("🎤 Listening for motion and providing automatic guidance")
        print("📷 Press 'q' to quit, 's' to force scan")
        
        self.speak_async("Automatic navigation assistant ready. Start moving for guidance.")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam")
                break
            
            frame_count += 1
            
            # Process every few frames for better performance
            if frame_count % 3 != 0:
                cv2.imshow("Auto Navigation Assistant - 'q' to quit, 's' to scan", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # Detect motion
            is_moving = self.detect_motion(frame)
            
            # Check if we should perform a scan
            if self.should_scan(is_moving) or cv2.waitKey(1) & 0xFF == ord('s'):
                self.last_scan_time = time.time()
                
                # Process frame for navigation
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Object detection
                results = self.yolo_model(frame_rgb)
                detections = results.xyxy[0].cpu().numpy()
                
                # Depth estimation
                input_tensor = self.midas_transform(frame_rgb).to(self.device)

                # input_tensor should have shape [1, 3, H, W] — check this!
                with torch.no_grad():
                    prediction = self.midas(input_tensor)  # Output: (1, 1, h, w) or (1, h, w)

                    if prediction.dim() == 3:
                        prediction = prediction.unsqueeze(1)  # Make it (1, 1, H, W)

                    prediction = torch.nn.functional.interpolate(
                        prediction,
                        size=frame.shape[:2],  # (H, W)
                        mode="bicubic",
                        align_corners=False,
                    )

                depth_map = prediction.squeeze().cpu().numpy()

                
                # Generate guidance
                urgent_warnings, important_obstacles, general_info = self.generate_navigation_guidance(
                    frame, detections, depth_map
                )
                
                # Determine what to announce
                priority, announcement_list = self.should_announce(
                    urgent_warnings, important_obstacles, general_info, is_moving
                )
                
                if priority and announcement_list:
                    self.last_announcement_time = time.time()
                    
                    announcement = ". ".join(announcement_list)
                    
                    # Add motion context for better user awareness
                    if priority == "urgent":
                        print(f"🚨 URGENT: {announcement}")
                        self.speak_async(announcement, priority="urgent")
                    else:
                        motion_status = "moving" if is_moving else "stationary"
                        print(f"🧭 [{motion_status.upper()}] {announcement}")
                        self.speak_async(announcement)
            
            # Show frame with minimal info
            status_text = "MOVING" if is_moving else f"STILL ({self.stationary_time:.1f}s)"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Auto Navigation Assistant - 'q' to quit, 's' to scan", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Automatic navigation assistant stopped.")
        self.speak_async("Navigation assistant stopped. Stay safe!")

# Run the automatic navigation assistant
if __name__ == "__main__":
    assistant = AutoNavigationAssistant()
    assistant.run()