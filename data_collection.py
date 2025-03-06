import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize video capture and hand detector
cap = cv2.VideoCapture(1)  # Use 1 or 2 if the wrong camera is selected
detector = HandDetector(maxHands=2, detectionCon=0.7)  # Detect up to 2 hands with high confidence

# Image processing settings
offset = 20
imgSize = 300
counter = 0
capture_interval = 0.5  # Capture image every 0.5 seconds
last_capture_time = time.time()

# Ensure the folder exists
folder = r"C:\Users\Lenovo\Desktop\sign language detection\Data\Sit"
os.makedirs(folder, exist_ok=True)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Camera not detected. Exiting...")
    exit()

while True:
    success, img = cap.read()

    if not success:
        print("Error: Could not read frame. Restarting camera...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(0)
        continue

    hands, img = detector.findHands(img, draw=True, flipType=True)  # Enable flipping for correct hand identification

    if hands:
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0

        for hand in hands:
            handType = hand["type"]  # 'Left' or 'Right'
            x, y, w, h = hand['bbox']
            print(f"Detected {handType} Hand at {x, y, w, h}")

            x_min = min(x_min, x - offset)
            y_min = min(y_min, y - offset)
            x_max = max(x_max, x + w + offset)
            y_max = max(y_max, y + h + offset)

        x1, y1 = max(0, x_min), max(0, y_min)
        x2, y2 = min(img.shape[1], x_max), min(img.shape[0], y_max)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("Warning: Cropped image is empty. Skipping frame.")
        else:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # White background

            h, w, _ = imgCrop.shape
            aspectRatio = h / w if w > 0 else 1  # Prevent division by zero

            if aspectRatio > 1:  # Tall Image
                k = imgSize / h
                wCal = max(1, math.ceil(k * w))
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:  # Wide Image
                k = imgSize / w
                hCal = max(1, math.ceil(k * h))
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

            # Automatically capture images at intervals
            if time.time() - last_capture_time >= capture_interval:
                counter += 1
                imgPath = os.path.join(folder, f'Image_{counter}.jpg')
                cv2.imwrite(imgPath, imgWhite)
                last_capture_time = time.time()
                print(f"Saved: {imgPath} (Count: {counter})")

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
