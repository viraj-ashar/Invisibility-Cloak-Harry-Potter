import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(3)
background=0

for i in range(30):
    ret,background = cap.read()

background = np.flip(background,axis=1)

while(cap.isOpened()):
    ret, img = cap.read()

    # Flip the image
    img = np.flip(img, axis = 1)

    # Convert the image to HSV color space.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (35, 35), 0)

    # Defining lower range for red color detection.
    lower = np.array([70,77,64])
    upper = np.array([135,255,255])
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # Replacing pixels corresponding to cloak with the background pixels.
    img[np.where(mask == 255)] = background[np.where(mask == 255)]
    cv2.imshow('Display',img)
    k = cv2.waitKey(1)
    if k == 13:
        break