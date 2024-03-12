
import cv2
import numpy as np
import pyautogui

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the fixed size for the bounding box (e.g., the size of a finger)
BOX_SIZE = 50

# Function to detect the finger based on skin color and draw a fixed-size box
def detect_finger_and_draw_box(frame):
    # Convert frame to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for skin color
    # Note: These values may need adjustment depending on the skin tone and lighting conditions
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for the skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour, assumed to be the hand
        max_contour = max(contours, key=cv2.contourArea)
        
        # Find the convex hull of the hand
        hull = cv2.convexHull(max_contour)
        
        # Find the topmost point of the hull, which should correspond to the tip of the finger
        topmost = tuple(hull[hull[:, :, 1].argmin()][0])
        
        # Draw a fixed-size box around the tip of the finger
        top_left = (topmost[0] - BOX_SIZE // 2, topmost[1] - BOX_SIZE // 2)
        bottom_right = (topmost[0] + BOX_SIZE // 2, topmost[1] + BOX_SIZE // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        
        # Return the center of the box instead of the tip for smoother movement
        box_center = (topmost[0], topmost[1])
        
        return frame, box_center
    
    return frame, None

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame for finger detection
    processed_frame, finger_position = detect_finger_and_draw_box(frame)
    
    # Only proceed if a finger_position was detected
    if finger_position:
        # Show the processed frame
        cv2.imshow('Finger Tracking', processed_frame)
        
        # Map the finger position to the screen size
        screen_width, screen_height = pyautogui.size()
        mapped_x = np.interp(finger_position[0], (0, frame.shape[1]), (0, screen_width))
        mapped_y = np.interp(finger_position[1], (0, frame.shape[0]), (0, screen_height))
        
        # Move the mouse cursor to the mapped position
        pyautogui.moveTo(mapped_x, mapped_y)
        
        # Print the coordinates in the terminal
        print(f'Cursor Position: X: {mapped_x}, Y: {mapped_y}')
    else:
        print("Finger not detected")
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
