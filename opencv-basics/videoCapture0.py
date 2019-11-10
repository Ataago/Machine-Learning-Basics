# 
#   videoCapture0.py
#   Capturing first Frame
#
#   Created by Mohammed Ataa on 10/11/19.
#   Copyright © 2019 Ataago. All rights reserved.
#   

import cv2
import time

# Turn on Camera
video = cv2.VideoCapture(0)

# First Frame is captured .. camera is on for 1 sec then released
check, frame = video.read()
time.sleep(1)
video.release()

# Display the Captured image
cv2.imshow("Capture", frame)
cv2.waitKey(1000)
cv2.destroyAllWindows()