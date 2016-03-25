import cv2

from FaceDetection import FaceDetection

from CameraStream import CameraStream

from multiprocessing.pool import ThreadPool
from collections import deque
import time
from common import clock, draw_str, StatValue
import copy

class ProcImg():
    def __init__(self, stream):
        self.stream = stream
        #self.numThread = cv2.getNumberOfCPUs()
        self.numThread = 5
        self.workerPool = ThreadPool(processes = self.numThread)
        self.pendingWorker = deque()

        self.latency = StatValue()
        self.frameInterval = StatValue()
        self.lastFrameTime = clock()

        self.outFrames = deque(maxlen = 5)

    def isEmpty(self):
        return(len(self.outFrames) == 0)

    def getFrame(self):
        return self.outFrames.popleft()
            
    def threadedProcess(self):
              
    
        if len(self.pendingWorker) > 0 and self.pendingWorker[0].ready():
            task = self.pendingWorker.popleft()
            frame, time = task.get()
            self.latency.update(clock() - time)
            draw_str(frame, (20, 20), "Latency: %.1f ms" % (self.latency.value*1000))
            draw_str(frame, (300, 20), "FPS: %d" % (1/self.frameInterval.value))
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

def process(frame, t):
    #Face Detection    
    numFaces, rects = FaceDetection(frame).run() 
    #Face Recognition 

    return frame, t

if __name__ == '__main__':
    from imutils.video.pivideostream import PiVideoStream
    stream = CameraStream(resolution = (640, 480)).start()
    time.sleep(2.0)
    imgProc = ProcImg(stream)
    while True:
        imgProc.threadedProcess()
        if not imgProc.isEmpty():
            frame = imgProc.getFrame()
            cv2.imshow('Processed Frame', frame)
            cv2.waitKey(1)
