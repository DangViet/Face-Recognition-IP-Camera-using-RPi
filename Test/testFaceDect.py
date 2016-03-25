import numpy as np
import cv2
import imutils.video import FPS
import imutils

face_cascade = cv2.CascadeClassifier('/usr/local/lib/python2.7/dist-packages/SimpleCV/Features/HaarCascades/face.xml')

img = cv2.imread('kairos-elizabeth.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
	print 'Got a face'
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
small = cv2.resize(img, (0,0), fx=0.25, fy=0.25)     
cv2.imshow('small',small)
cv2.waitKey(0)
cv2.destroyAllWindows()
