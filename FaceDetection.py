import cv2
 
class FaceDetection():
    def __init__(self, image):
        self.image = image         # OpenCV image array
        self.drawn = 0          # Count of how many detector-boxes have been drawn
        self.path = "/home/pi/HaarCascades/"      # The path to the haarcascades data xml files
        self.rects = []         # Discovered rectangles from Image Analysis
        
        self.minSize = (10, 10)
        self.imageScale = 1.5
        self.haar_scale = 1.2
        self.min_neighbors = 2
        self.haarFlags = cv2.cv.CV_HAAR_DO_CANNY_PRUNING
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
 
    def detect(self, xml):          # Detect people in image and save the image bounds around them
        cascade = cv2.CascadeClassifier(self.path + xml)
        if cascade == None:
            print 'Cannot load Cascade Classifier'
        #self.image = cv2.imread(self.image_name)  # Loads the image into a numpy array
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        smallImg = cv2.resize(gray, (0,0), fx = 1/float(self.imageScale), fy = 1/float(self.imageScale))
        #cv2.equalizeHist(smallImg, smallImg)
        hits = cascade.detectMultiScale(      # Grayscale and analyze image using 
            smallImg,                 #  the selected cascade classifier file
            scaleFactor=self.haar_scale,
            minNeighbors=self.min_neighbors,
            minSize= self.minSize,
            flags= self.haarFlags
        )
        self.rects.append(hits)     # Add detected people to rect-list for drawing
        return hits         # Can use len(hits) to check if anyone was found
 
    # The following functions provide an xml file to be used as the cascade classifier
    #  to detect different things, such as face, upper body, or pedestrian
    def face(self):
        return self.detect('haarcascade_frontalface_alt.xml')
 
    def face2(self):
        return self.detect('lbpcascade_frontalface.xml')
 
    def face3(self):
        return self.detect('face.xml')
 
    def full_body(self):
        return self.detect('haarcascade_fullbody.xml')
 
    def upper_body(self):
        return self.detect('haarcascade_upperbody.xml')
 
    def pedestrian(self):
        return self.detect("haarcascade_pedestrians.xml")
 
    # This function will draw the rectangles around all objects found and then 
    #  overwrite the original image file.
    def draw(self):
        for hits in self.rects:
            color = (0, 0, 255)# Rect color selection
            self.drawn += 1
            for (x,y,w,h) in hits:
                pt1 = (int(x * self.imageScale), int(y * self.imageScale))
                pt2 = (int((x + w) * self.imageScale), int((y + h) * self.imageScale))
                cv2.rectangle(self.image,  pt1, pt2, color, 2) # Draws the Rect
        return hits
    def run(self):
        numFaces = len(self.face2())
        #faces += len(self.face2())
        #faces += len(self.face3())
        #uppers = len(self.upper_body())
        #fulls = len(self.full_body())
        #peds = len(self.pedestrian())
        #self.draw()         
        rects = []
        for hits in self.rects:
            for (x, y, w, h)in hits:
                rects.append((self.imageScale*x, self.imageScale*y,self.imageScale*w, self.imageScale*h))
        #self.overlay()
   
        return numFaces, rects
