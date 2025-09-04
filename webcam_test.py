import cv2

def stream_webcam():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit the webcam stream.")
    
    # Stream from webcam
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Display the resulting frame
        cv2.imshow('Webcam Stream', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_webcam()
