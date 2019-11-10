# 
#   videoCapture1.py
#   Capturing a video
#
#   Created by Mohammed Ataa on 10/11/19.
#   Copyright © 2019 Ataago. All rights reserved.
#   

import cv2
import time

# Turn on Camera
video = cv2.VideoCapture(0)

# Capture multiple Frames to make a video
frame_count = 1

while True:
    frame_count += 1

    check, frame = video.read()
    cv2.imshow('Capture', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("Number of Frames captured: ", frame_count)


