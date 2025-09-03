import cv2
import argparse
import sys
import os
import numpy as np
import time

# Constants for movement thresholds
HORIZ_THRESHOLD = 0.15  # 15% of frame width for left/right movement
VERT_THRESHOLD = 0.05   # 5% of frame height for jump/duck movement

class MovementDetector:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.reference_x = None
        self.reference_y = None
        self.reference_height = None
        self.reference_width = None
        
        # Movement thresholds in pixels
        self.horiz_threshold = int(frame_width * HORIZ_THRESHOLD)
        self.vert_threshold = int(frame_height * VERT_THRESHOLD)
        
        # For filtering out jitter
        self.horiz_history = []
        self.vert_history = []
        self.history_size = 3
        
    def set_reference(self, center_x, center_y, width, height):
        """Set reference position for the detected person"""
        self.reference_x = center_x
        self.reference_y = center_y
        self.reference_width = width
        self.reference_height = height
        print(f"Reference position set: ({center_x}, {center_y}), size: {width}x{height}")
        
    def detect_movement(self, center_x, center_y, width, height):
        """Detect movement based on current position compared to reference"""
        # Set reference position if this is the first detection
        if self.reference_x is None:
            self.set_reference(center_x, center_y, width, height)
            return "middle+standing"
        
        # Determine horizontal movement (left/middle/right)
        horizontal_diff = center_x - self.reference_x
        vertical_diff = center_y - self.reference_y
        height_ratio = height / self.reference_height if self.reference_height > 0 else 1
        
        # Classify horizontal position
        if horizontal_diff < -self.horiz_threshold:
            h_position = "left"
        elif horizontal_diff > self.horiz_threshold:
            h_position = "right"
        else:
            h_position = "middle"
            
        # Classify vertical action
        if vertical_diff < -self.vert_threshold:
            v_action = "jump"
        elif vertical_diff > self.vert_threshold or height_ratio < 0.8:
            v_action = "duck"
        else:
            v_action = "standing"
            
        # Add to history for smoothing
        self.horiz_history.append(h_position)
        self.vert_history.append(v_action)
        
        if len(self.horiz_history) > self.history_size:
            self.horiz_history.pop(0)
        if len(self.vert_history) > self.history_size:
            self.vert_history.pop(0)
            
        # Get most common movements in history
        if len(self.horiz_history) >= self.history_size and len(self.vert_history) >= self.history_size:
            # Count horizontal positions
            h_counts = {}
            for pos in self.horiz_history:
                if pos not in h_counts:
                    h_counts[pos] = 0
                h_counts[pos] += 1
            
            # Count vertical actions
            v_counts = {}
            for act in self.vert_history:
                if act not in v_counts:
                    v_counts[act] = 0
                v_counts[act] += 1
            
            # Find the most common positions
            most_common_h = max(h_counts.items(), key=lambda x: x[1])
            most_common_v = max(v_counts.items(), key=lambda x: x[1])
            
            if most_common_h[1] >= self.history_size // 2 and most_common_v[1] >= self.history_size // 2:
                h_position = most_common_h[0]
                v_action = most_common_v[0]
        
        # Combine horizontal position and vertical action
        return f"{h_position}+{v_action}"

def load_model():
    """
    Load the pre-trained MobileNet-SSD model for object detection.
    
    Returns:
        net: The neural network model
        classes: List of object classes the model can detect
    """
    # Path to the model files
    # These paths work with OpenCV's sample models
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Create the models directory if it doesn't exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Paths for the model files
    prototxt_path = os.path.join(model_path, "MobileNetSSD_deploy.prototxt")
    model_weights = os.path.join(model_path, "MobileNetSSD_deploy.caffemodel")
    
    # Check if model files exist, if not inform user to download them
    if not os.path.exists(prototxt_path) or not os.path.exists(model_weights):
        print("Required model files not found. Please download them from:")
        print("https://github.com/chuanqi305/MobileNet-SSD/")
        print(f"Place the files in: {model_path}")
        return None, None
    
    # Load the model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_weights)
    
    # Define the classes the model can detect
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
              "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
              "horse", "motorbike", "person", "pottedplant", "sheep", 
              "sofa", "train", "tvmonitor"]
    
    return net, classes

def detect_humans(frame, net, classes, movement_detector):
    """
    Detect humans in a frame, track their movement, and draw bounding boxes.
    
    Args:
        frame: Input video frame
        net: Neural network model
        classes: List of object classes
        movement_detector: MovementDetector object to track movements
    
    Returns:
        frame: Frame with bounding boxes and movement annotations
    """
    if net is None or classes is None:
        return frame
        
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    
    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # Set the blob as input to the network
    net.setInput(blob)
    
    # Get detections
    detections = net.forward()
    
    # Track highest confidence person detection
    best_confidence = 0
    best_box = None
    
    # Process detections
    for i in range(detections.shape[2]):
        # Get the confidence of the detection
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Get the class ID
            class_id = int(detections[0, 0, i, 1])
            
            # Check if the detected object is a person (class_id 15)
            if class_id == 15 and confidence > best_confidence:  # person with highest confidence
                best_confidence = confidence
                best_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    
    # Process the best person detection
    if best_box is not None:
        (startX, startY, endX, endY) = best_box.astype("int")
        
        # Calculate center and dimensions of the bounding box
        center_x = (startX + endX) // 2
        center_y = (startY + endY) // 2
        box_width = endX - startX
        box_height = endY - startY
        
        # Detect movement - returns combined classification like "left+jump"
        movement = movement_detector.detect_movement(center_x, center_y, box_width, box_height)
        
        # Parse the combined movement classification
        if "+" in movement:
            h_position, v_action = movement.split("+")
        else:
            h_position, v_action = "middle", "standing"
        
        # Set color based on horizontal position
        h_color_map = {
            "left": (255, 0, 0),     # Blue
            "middle": (0, 255, 0),   # Green
            "right": (0, 255, 255),  # Yellow
        }
        
        # Set intensity based on vertical action
        v_intensity_map = {
            "jump": 1.0,    # Full brightness
            "standing": 0.7,  # Medium brightness
            "duck": 0.4     # Lower brightness
        }
        
        # Get base color from horizontal position
        base_color = h_color_map.get(h_position, (0, 255, 0))  
        
        # Adjust color intensity based on vertical action
        intensity = v_intensity_map.get(v_action, 0.7)
        box_color = tuple(int(c * intensity) for c in base_color)
        
        # Draw the bounding box
        cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
        
        # Draw reference position (if set)
        if movement_detector.reference_x is not None:
            cv2.circle(frame, (movement_detector.reference_x, movement_detector.reference_y), 
                       5, (0, 0, 255), -1)  # Red dot for reference
        
        # Add labels
        confidence_label = f"Person: {best_confidence:.2f}"
        movement_label = f"Movement: {movement}"
        print(movement)
        
        # Display labels
        y1 = startY - 35 if startY > 35 else startY + 35
        y2 = startY - 15 if startY > 15 else startY + 15
        cv2.putText(frame, confidence_label, (startX, y1), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        cv2.putText(frame, movement_label, (startX, y2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    return frame

def play_video(video_path):
    """
    Play an MP4 video file using OpenCV with human detection and movement tracking.
    
    Args:
        video_path (str): Path to the MP4 file
    """
    # Load the object detection model
    net, classes = load_model()
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Print video information
    print(f"Video resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    
    # Create movement detector
    movement_detector = MovementDetector(frame_width, frame_height)
    
    # Create a window to display the video
    window_name = "Video Player with Combined Movement Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # For movement tracking display
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # white
    line_type = 2
    
    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        frame_count += 1
        if frame_count % 2 == 0:
            continue
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
            
        # Detect humans in the frame, track movement, and draw bounding boxes
        if net is not None and classes is not None:
            frame = detect_humans(frame, net, classes, movement_detector)
        
        # Add combined movement classification explanation
        cv2.putText(frame, "Movement Classification: [Horizontal]+[Vertical]", 
                   (10, 30), font, font_scale, font_color, line_type)
        cv2.putText(frame, "Horizontal: left, middle, right", 
                   (10, 60), font, font_scale, font_color, line_type)
        cv2.putText(frame, "Vertical: standing, jump, duck", 
                   (10, 90), font, font_scale, font_color, line_type)
                   
        # Add reference position and controls information
        if movement_detector.reference_x is not None:
            cv2.putText(frame, "Red dot = Reference Position | Press 'r' to reset reference", 
                       (10, 120), font, 0.7, (0, 0, 255), line_type)
        
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Reset reference position if 'r' is pressed
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == ord('r'):
            movement_detector.reference_x = None
            movement_detector.reference_y = None
            movement_detector.reference_width = None
            movement_detector.reference_height = None
            print("Reference position reset. Next detection will set a new reference.")
        # Exit if 'q' is pressed or window is closed
        elif key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Play an MP4 video file using OpenCV.')
    parser.add_argument('video_path', type=str, help='Path to the MP4 file')
    args = parser.parse_args()
    
    # Play the video
    play_video(args.video_path)

if __name__ == "__main__":
    main()
