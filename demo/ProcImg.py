import cv2
from subprocess import call  
import subprocess
from datetime import datetime
from FaceDetection import FaceDetection
from FaceDetection1 import FaceDetection1
import config
from multiprocessing.pool import ThreadPool
from collections import deque
import time
from common import clock, draw_str, StatValue
import copy
import FaceTracking
from Queue import Queue
import threading
class ProcImg():
    def __init__(self, stream):
        self.stream = stream
        self.numThread = cv2.getNumberOfCPUs()
        #self.numThread = 1
        self.workerPool = ThreadPool(processes = self.numThread)
        self.pendingWorker = deque()

        self.latency = StatValue()
        self.frameInterval = StatValue()
        self.lastFrameTime = clock()

        self.outFrames = deque(maxlen = self.numThread)
        self.faces = []

    def isEmpty(self):
        return(len(self.outFrames) == 0)

    def getFrame(self):
        return self.outFrames.popleft()
    def draw(self, rects, frame):
        
        color = (0, 0, 255)# Rect color selection
        for (x,y,w,h) in rects:
            pt1 = (int(x), int(y))
            pt2 = (int((x+w)), int(y + h))
            cv2.rectangle(frame,  pt1, pt2, color, 1) # Draws the Rect
        #return hits

        
    def threadedProcess(self):
              
        rects = [] 
        if len(self.pendingWorker) > 0 and self.pendingWorker[0].ready():
            task = self.pendingWorker.popleft()
            frame, curTime = task.get()
            self.latency.update(clock() - curTime)
            
            draw_str(frame, (20, 360-20), "Latency: %.1f ms" % (self.latency.value*1000))
            draw_str(frame, (20, 360- 35), "FPS: %d" % (1/self.frameInterval.value))
            self.outFrames.append(frame)
        '''
        if len(self.pendingWorker) > 0:
            for i in range(0, len(self.pendingWorker)):
                if self.pendingWorker[i].ready():
                    for j in range(0, i):
                        waste = self.pendingWorker.popleft()
                        try:
                            waste.get()
                        except:
                            pass

                    task = self.pendingWorker.popleft()
                    frame, time = task.get()
                    self.latency.update(clock() - time)
                    draw_str(frame, (20, 20), "Latency: %.1f ms" % (self.latency.value*1000))
                    draw_str(frame, (300, 20), "FPS: %d" % (1/self.frameInterval.value))
                    cv2.imshow('Processed Video', frame)
                    cv2.waitKey(1)
                    break
        '''
        if len(self.pendingWorker) < self.numThread:
            grab, frame = self.stream.read()
            t = clock()
            self.frameInterval.update(t - self.lastFrameTime)
            self.lastFrameTime = t
            task = self.workerPool.apply_async(process, (copy.copy(frame), t))
            self.pendingWorker.append(task)
        


    def stop(self):
        workerPool.terminate()
        pendingWorker.clear()
def draw(rects, mode, method,frame):
    color = (0, 0, 255)# Rect color selection
    for i in range(0, len(rects)):
        (x,y,w,h) = rects[i]
        pt1 = (int(x), int(y))
        pt2 = (int((x+w)), int(y + h))
        cv2.rectangle(frame,  pt1, pt2, color, 1) # Draws the Rect

    draw_str(frame, (20, 360 - 65),"Frame Num: %d"%(mode))   
    draw_str(frame, (20, 360- 50),"Mode: %s"%(method))
    #draw_str(frame, (20, config.VIDEO_WIDTH - 50),time.ctime())
faces = []
lockGlobVar = threading.Lock()
mode = 0
allRoiPts = []
allRoiHist = []
def process(frame, t):
    #global faces
    #global labels
    #global conf
    global mode 
    global lockGlobVar
    global allRoiPts
    global allRoiHist
    #Face Detection    
    if mode < 50:
        numRoi, allRoiPts= FaceDetection1(frame).run() 
        #lockGlobVar.acquire()
        allRoiHist = FaceTracking.calcHist(frame, allRoiPts)
        draw(allRoiPts, mode, "Haar-like Face Detection",frame)
        mode +=1
        #lockGlobVar.release()
    elif mode >=50 and mode < 150:
        numRoi, allRoiPts= FaceDetection(frame).run()
        lockGlobVar.acquire()
        allRoiHist = FaceTracking.calcHist(frame, allRoiPts)
        draw(allRoiPts, mode, "LBP Face Detection",frame)
        mode +=1
        lockGlobVar.release()

    else:
        #Face Tracking 
        #faces = FaceTracking(frame, lastFrame, faces)
        #labels, conf = faceRecognition(frame, faces)
    #draw(faces, labels, conf, frame)
        if len(allRoiPts) != 0:
            lockGlobVar.acquire()
            _allRoiPts = copy.copy(allRoiPts)
            _allRoiHist = copy.copy(allRoiHist)
            mode +=1
            allRoiPts = FaceTracking.Track(frame, _allRoiPts, _allRoiHist)
            draw(copy.copy(allRoiPts), mode, "Face Tracking",frame)
            
            lockGlobVar.release()

    return frame, t

if __name__ == '__main__':
    #from imutils.video.pivideostream import PiVideoStream
    #stream = WebcamVideoStream().start()
    #time.sleep(0.5)
    #imgProc = ProcImg(stream)
    try:
        while True:
            imgProc.threadedProcess()
            if not imgProc.isEmpty():
                frame = imgProc.getFrame()
                cv2.imshow('Processed Frame', frame)
                cv2.waitKey(1)
    except KeyboardInterrupt:
        stream.stop()
        cv2.destroyAllWindows()
