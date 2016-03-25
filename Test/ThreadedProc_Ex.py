#!/usr/bin/env python

'''
Multithreaded video processing sample.
Usage:
   video_threaded.py {<video device number>|<video file name>}
   Shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.
Keyboard shortcuts:
   ESC - exit
   space - switch between multi and single threaded processing
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

from multiprocessing.pool import ThreadPool
from collections import deque

from common import clock, draw_str, StatValue
import video

from picamera.array import PiRGBArray
from picamera import PiCamera

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils

import copy
class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data

if __name__ == '__main__':
    import sys

    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0
    '''camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCap= PiRGBArray(camera, size = (640, 480))'''
    cap =  PiVideoStream().start()


    def process_frame(frame, t0):
        # some intensive computation...
        frame = cv2.medianBlur(frame, 19)
        # frame = cv2.medianBlur(frame, 19)
        return frame, t0

    threadn = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    threaded_mode = True

    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()
    while True:
        while len(pending) > 0 and pending[0].ready():
            '''   '''
            res, t0 = pending.popleft().get()
            latency.update(clock() - t0)
            draw_str(res, (20, 20), "Latency: %.1f ms" % (latency.value*1000))
            draw_str(res, (100, 20), "Frame interval: %.1f ms" % (frame_interval.value*1000))
            print('Interval: %.lf ms',(frame_interval.value*1000))
            #cv2.imshow('threaded video', res)
            frame = cv2.medianBlur(frame, 19)
        if len(pending) < threadn:
            #camera.capture(rawCap, format = "bgr")
            #frame = rawCap.array
            frame = cap.read()
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (copy.copy(frame), t))
                #task = pool.apply_async(process_frame, (frame, t))
            else:
                task = DummyTask(process_frame(frame, t))
            #rawCap.truncate(0)
            pending.append(task)
        ch = 0xFF & cv2.waitKey(1)
        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break
cv2.destroyAllWindows()
