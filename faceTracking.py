import cv2
import cv2.cv as cv
import numpy as np

class FaceTracking():
    curFrame = []
    lastFrame = []
    def __init__(self, face):
        self.firstRun = True
        self.rect = face
        self.VMin = 
        self.VMax =
        self.SMin = 
        self.trackWindow = []
        self.hsize = None
        self.hsv = None
        self.mask = None
        self.hist = None
        self.backProj = None
        self.origin = None
        self.selection = None 
        
