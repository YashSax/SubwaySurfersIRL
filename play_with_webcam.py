import cv2
import os
import numpy as np
import subprocess
from typing import Tuple, List, Dict, Optional, Union, Any
import time
import numpy.typing as npt
from art import tprint
import time

# Import MovementDetector class from subway_surfers.py
from subway_surfers import MovementDetector, PositionSmoother, load_model, detect_humans

# Constants for movement thresholds
HORIZ_BOX_THRESHOLD = 0.15  # 35% of initial bounding box width for left/right movement
JUMP_BOX_THRESHOLD = 0.05   # 5% of initial bounding box height for jump detection
DUCK_BOX_THRESHOLD = 0.15   # 15% of initial bounding box height for duck detection

# ADB Control functions
class ADBController:
    def __init__(self):
        # Start persistent adb shell
        self.adb = subprocess.Popen(
            ["adb", "shell"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
    
    def send_cmd(self, cmd: str):
        """Send a command to the persistent adb shell."""
        self.adb.stdin.write(cmd + "\n")
        self.adb.stdin.flush()

    def jump(self):
        """Perform jump action (swipe up)"""
        print("Executing Jump")
        self.send_cmd("input swipe 500 1200 500 600 3")   # fast upward swipe

    def roll(self):
        """Perform roll/duck action (swipe down)"""
        print("Executing Roll")
        self.send_cmd("input swipe 500 600 500 1200 3")   # fast downward swipe

    def left(self):
        """Perform left movement (swipe left)"""
        print("Executing Left Swipe")
        self.send_cmd("input swipe 800 1000 300 1000 3")  # fast left swipe

    def right(self):
        """Perform right movement (swipe right)"""
        print("Executing Right Swipe")
        self.send_cmd("input swipe 300 1000 800 1000 3")  # fast right swipe

class GameController:
    def __init__(self):
        self.adb_controller = ADBController()
        self.prev_h_position = "middle"
        self.prev_v_action = "standing"
        
    def handle_movement_change(self, h_position: str, v_action: str):
        print("HEREEEEEEEEE")
        print("Positions:", h_position, v_action, self.prev_h_position, self.prev_v_action)
        """
        Handle movement changes and translate them to game controls
        
        Rules:
        - middle to left: swipe left once
        - middle to right: swipe right once
        - left to right: swipe right twice
        - right to left: swipe left twice
        - left to middle: swipe right once
        - right to middle: swipe left once
        - standing to jump: swipe up once
        - standing to duck: swipe down once
        - jump to duck: swipe down twice
        - duck to jump: swipe up twice
        - No actions for jump to standing or duck to standing
        """
        # Handle horizontal movement changes
        if h_position != self.prev_h_position:
            if self.prev_h_position == "middle" and h_position == "left":
                self.adb_controller.left()
            elif self.prev_h_position == "middle" and h_position == "right":
                self.adb_controller.right()
            elif self.prev_h_position == "left" and h_position == "right":
                self.adb_controller.right()
                time.sleep(0.1)  # Small delay between commands
                self.adb_controller.right()
            elif self.prev_h_position == "right" and h_position == "left":
                self.adb_controller.left()
                time.sleep(0.1)  # Small delay between commands
                self.adb_controller.left()
            elif self.prev_h_position == "left" and h_position == "middle":
                self.adb_controller.right()
            elif self.prev_h_position == "right" and h_position == "middle":
                self.adb_controller.left()
        
        # Handle vertical movement changes
        if v_action != self.prev_v_action:
            if self.prev_v_action == "standing" and v_action == "jump":
                self.adb_controller.jump()
            elif self.prev_v_action == "standing" and v_action == "duck":
                self.adb_controller.roll()
            elif self.prev_v_action == "jump" and v_action == "duck":
                self.adb_controller.roll()
                time.sleep(0.1)  # Small delay between commands
                self.adb_controller.roll()
            elif self.prev_v_action == "duck" and v_action == "jump":
                self.adb_controller.jump()
                time.sleep(0.1)  # Small delay between commands
                self.adb_controller.jump()
                
            # Skip jump to standing and duck to standing transitions
        
        # Update previous positions
        self.prev_h_position = h_position
        self.prev_v_action = v_action

def play_with_webcam():
    """
    Stream webcam feed, detect human movements, and control Subway Surfers game via ADB
    """
    # Load the pre-trained model for human detection
    net, classes = load_model()
    if net is None or classes is None:
        print("Failed to load detection model.")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize movement detector with frame dimensions
    movement_detector = MovementDetector(frame_width, frame_height)
    smoother = PositionSmoother()
    
    # Initialize game controller
    game_controller = GameController()
    
    print("Webcam controller ready!")
    print("Press 'q' to quit or 'r' to recalibrate.")
    
    # Variables to store current position and action
    current_movement = "middle+standing"
    
    # Stream from webcam
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Flip the frame horizontally for a more natural mirror view
        frame = cv2.flip(frame, 1)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        
        # Press 'r' to recalibrate
        elif key == ord('r'):
            print("Recalibrating...")
            movement_detector.reference_x = None
            movement_detector.reference_y = None
            movement_detector.reference_width = None
            movement_detector.reference_height = None
            movement_detector.horiz_history = []
            movement_detector.vert_history = []
        
        # Run human detection on the frame
        processed_frame, new_movement = detect_humans(frame, net, classes, movement_detector, smoother)
        
        # Get movement detection results
        if new_movement:
            # Add overlay with current movement
            cv2.putText(
                processed_frame, 
                new_movement, 
                (50, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                5, 
                (0, 0, 0), 
                5
            )
        
        # Display the frame
        cv2.imshow('Subway Surfers Controller', processed_frame)

        # Only process if movement has changed
        if new_movement and new_movement != current_movement:
            print(f"Old movement: {current_movement}, new movement: {new_movement}")
            # Parse the movement string
            if "+" in new_movement:
                h_position, v_action = new_movement.split("+")
                # Handle movement change via game controller
                print(f"Prev: {game_controller.prev_h_position}, {game_controller.prev_v_action}, New: {h_position}, {v_action}")
                game_controller.handle_movement_change(h_position, v_action)
            current_movement = new_movement
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TIMER = 1
    for i in range(TIMER, 0, -1):
        print(f"{i}...")
        tprint(str(i), font="block")
        time.sleep(1)
    
    play_with_webcam()
