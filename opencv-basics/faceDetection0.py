# 
#   faceDetection0.py
#   Detecting a face
#
#   Created by Mohammed Ataa on 10/11/19.
#   Copyright © 2019 Ataago. All rights reserved.
#   
### Face Detection
#   1. Create a Cascade Classifier (contains Features of a face)
#   2. Search for coordinates of the face
#   3. Draw a box around that face coordinates
# 

import cv2

filename = 'img1.jpg'
# Read the Image
img = cv2.imread('opencv-basics/images/img1.jpg')
print(img.shape)

# Display the Image and display for 1 sec
cv2.imshow('My Image', img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Resize the Image and Display for 1 sec
resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Creating Cascade Classifier
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('opencv-basics/images/haarcascade_frontalface_default.xml')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Search for faces Coordinates
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

# Draw a box around the Face
for x, y, w, h in faces:
    print(x, y, w, h)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2) 

# Display the Image and display for 1 sec
cv2.imshow('My Image', img)
cv2.waitKey(1000)
cv2.destroyAllWindows()