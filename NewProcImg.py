import cv2
from subprocess import call  
import subprocess
from datetime import datetime
from FaceDetection import FaceDetection
import config
from WebcamStream import WebcamVideoStream
from CameraStream import CameraStream
from FaceRecognition import faceRecognition
from multiprocessing.pool import ThreadPool
from collections import deque
import time
from common import clock, draw_str, StatValue
import copy
import FaceTracking
from Queue import Queue
import internet
from NewFaceRecognition import NewFaceRecognition
import threading
import temp
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
            
            draw_str(frame, (20, config.VIDEO_WIDTH -20), "Latency: %.1f ms" % (self.latency.value*1000))
            draw_str(frame, (20, config.VIDEO_WIDTH - 35), "FPS: %d" % (1/self.frameInterval.value))
            #print("Latency %lf" % (self.latency.value*1000))
            #print("FPS: %d" % (1/self.frameInterval.value))
            self.outFrames.append(frame)
            #cv2.imshow('Processed Video', frame) 
            #cv2.waitKey(1)
        if len(self.pendingWorker) < self.numThread:
            frame = self.stream.read()
            t = clock()
            self.frameInterval.update(t - self.lastFrameTime)
            self.lastFrameTime = t
            task = self.workerPool.apply_async(process, (copy.copy(frame), t))
            self.pendingWorker.append(task)
        


    def stop(self):
        workerPool.terminate()
        pendingWorker.clear()
def draw(rects, labelAndConf, frame, color):
    #color = (0, 0, 255)# Rect color selection
    for i in range(0, len(rects)):
        (x,y,w,h) = rects[i]
        pt1 = (int(x), int(y))
        pt2 = (int((x+w)), int(y + h))
        label = labelAndConf[i][0]
        
        confidence = labelAndConf[i][1]
        cv2.rectangle(frame,  pt1, pt2, color, 5) # Draws the Rect
        if confidence <= 60 and confidence != 0:
            if label == 1:
                draw_str(frame, (int(x), int(y-5)), "-VietDang-%d"%(confidence))
            elif label == 2:
                draw_str(frame, (int(x), int(y-5)), "-NgoDat-%d"%(confidence))
        else:
            draw_str(frame, (int(x), int(y-5)), "--")
    draw_str(frame, (20, config.VIDEO_WIDTH - 50),time.ctime())

allRoiPts = []
allRoiHist = []
labels = []
conf = []
MODE= Queue()
MODE.put(-1)

lockGlobVar = threading.Lock()
def process(frame, t):
    #global faces
    #global labels
    #global conf
    global allRoiPts
    global allRoiHist
    global lockGlobVar
    allDrawPts = []
    mode = MODE.get()
    mode = (mode+1)%20
    MODE.put(mode)
    labelAndConf = []
    color = None
    if mode == 0: 
        #Face Detection    
        lockGlobVar.acquire()
        allRoiPts = []
        allRoiHist = []
        numRoi, allRoiPts= FaceDetection(frame).run() 
        for i in range(0, numRoi):
            labelAndConf.append((0, 0))
        allRoiHist = FaceTracking.calcHist(frame, allRoiPts)
        allDrawPts = allRoiPts
        lockGlobVar.release()
        color = (0,255,0)          
    else:
        if len(allRoiPts) != 0:
            lockGlobVar.acquire()
            _allRoiPts = copy.copy(allRoiPts)
            _allRoiHist = copy.copy(allRoiHist)
            lockGlobVar.release()
            allRoiPts = FaceTracking.Track(frame, _allRoiPts, _allRoiHist)        

            allDrawPts = allRoiPts
            #Face Recognition 
            labelAndConf = NewFaceRecognition(frame, copy.copy(allRoiPts))
            color = (0,255,255)
    draw(copy.copy(allDrawPts), copy.copy(labelAndConf), frame, color)

    '''
    if internet.On() and mode == 50:
        #upload to dropbox:
        cv2.imwrite('./temp.png', frame, [cv2.IMWRITE_PNG_COMPRESSION,9])
        photofile = "/home/pi/Dropbox-Uploader/dropbox_uploader.sh upload ./temp.png "+time.ctime().replace(" ", "_")+".png"   
        print photofile
        subprocess.Popen(photofile, shell=True)
    #else:
        #Face Tracking 
        #faces = FaceTracking(frame, lastFrame, faces)
        #labels, conf = faceRecognition(frame, faces)
    '''
    return frame, t

if __name__ == '__main__':
    stream = WebcamVideoStream().start()
    time.sleep(0.5)
    imgProc = ProcImg(stream)
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
