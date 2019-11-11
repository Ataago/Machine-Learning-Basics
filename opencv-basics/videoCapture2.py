# 
#   videoCapture2.py
#   Capturing a video with face detection
#
#   Created by Mohammed Ataa on 10/11/19.
#   Copyright © 2019 Ataago. All rights reserved.
#   

import cv2
import time

def detectFace(img):
    face_cascade = cv2.CascadeClassifier('opencv-basics/images/haarcascade_frontalface_default.xml')
    # Search for faces Coordinates
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=20)
    # Draw a box around the Face
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Turn on Camera
video = cv2.VideoCapture(0)

# Capture multiple Frames to make a video
frame_count = 1

while True:
    frame_count += 1

    check, frame = video.read()

    # Face Detection
    detectFace(frame)

    # Display Frame until 'q' is enterd
    cv2.imshow('Capture', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("Number of Frames captured: ", frame_count)


