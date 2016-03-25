
from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import time
import cv2

vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()
while True: 
    frame = vs.read()
 
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
 
 
cv2.destroyAllWindows()
vs.stop()
