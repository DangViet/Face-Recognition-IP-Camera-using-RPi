import cv2

from FaceDetection import FaceDetection

from CameraStream import CameraStream
from FaceRecognition import faceRecognition
from multiprocessing.pool import ThreadPool
from collections import deque
import time
from common import clock, draw_str, StatValue
import copy
from FaceTracking import FaceTracking

class ProcImg():
    def __init__(self, stream):
        self.stream = stream
        self.numThread = cv2.getNumberOfCPUs()
        #self.numThread = 5
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
            cv2.rectangle(frame,  pt1, pt2, color, 2) # Draws the Rect
        #return hits

        
    def threadedProcess(self):
              
        rects = [] 
        if len(self.pendingWorker) > 0 and self.pendingWorker[0].ready():
            task = self.pendingWorker.popleft()
            frame, time = task.get()
            self.latency.update(clock() - time)
            
            draw_str(frame, (20, 580), "Latency: %.1f ms" % (self.latency.value*1000))
            draw_str(frame, (20, 565), "FPS: %d" % (1/self.frameInterval.value))
            #print("Latency %lf" % (self.latency.value*1000))
            #print("FPS: %d" % (1/self.frameInterval.value))
            self.outFrames.append(frame)
            #cv2.imshow('Processed Video', frame) 
            #cv2.waitKey(1)
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
            frame = self.stream.read()
            t = clock()
            self.frameInterval.update(t - self.lastFrameTime)
            self.lastFrameTime = t
            task = self.workerPool.apply_async(process, (copy.copy(frame), t))
            self.pendingWorker.append(task)
        


    def stop(self):
        workerPool.terminate()
        pendingWorker.clear()
def draw(rects, labels, conf, frame):
    color = (0, 0, 255)# Rect color selection
    for i in range(0, len(rects)):
        (x,y,w,h) = rects[i]
        pt1 = (int(x), int(y))
        pt2 = (int((x+w)), int(y + h))
        #label = labels[i]
        #confidence = conf[i]
        cv2.rectangle(frame,  pt1, pt2, color, 2) # Draws the Rect
        #if label == 8:
        #    draw_str(frame, (int(x), int(y-5)), "-VietDang-%d"%(confidence))
        #else:
        #    draw_str(frame, (int(x), int(y-5)), "--%d"%(confidence))

lastFrame = None
faces = []
labels = []
conf = []
mode = 0
def process(frame, t):
    global mode
    global faces
    global labels
    global conf
    global mode
    global lastFrame
    if mode == 0 :
        #Face Detection    
        numFaces, faces = FaceDetection(frame).run() 
        #Face Recognition 
        labels, conf = faceRecognition(frame, faces)
    else:
        #Face Tracking 
        faces = FaceTracking(frame, lastFrame, faces)
        #labels, conf = faceRecognition(frame, faces)
    mode = (mode+1)%4
    print mode
    lastFrame = copy.copy(frame)
    draw(faces, labels, conf, frame)
    return frame, t

if __name__ == '__main__':
    from imutils.video.pivideostream import PiVideoStream
    stream = CameraStream(resolution = (640, 480)).start()
    time.sleep(1.0)
    imgProc = ProcImg(stream)
    while True:
        imgProc.threadedProcess()
        if not imgProc.isEmpty():
            frame = imgProc.getFrame()
            cv2.imshow('Processed Frame', frame)
            cv2.waitKey(1)
