
import cv2
import numpy as np
import pyautogui
from cursor import handDetector  # Ensure cursor.py is in the same directory or add it to the Python path

def cursor2():
    # Screen size for mouse movement scaling
    screenW, screenH = pyautogui.size()
    frameR = 100  # Frame reduction to create a smaller work area on the screen

    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            # Tip of the index finger
            x1, y1 = lmList[8][1], lmList[8][2]
            
            # Convert coordinates
            mouseX = np.interp(x1, (frameR, cap.get(3) - frameR), (0, screenW))
            mouseY = np.interp(y1, (frameR, cap.get(4) - frameR), (0, screenH))
            
            # Move the mouse
            pyautogui.moveTo(screenW - mouseX, mouseY)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




