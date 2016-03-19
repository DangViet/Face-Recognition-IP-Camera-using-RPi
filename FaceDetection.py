import cv2
 
class FaceDetection():
    def __init__(self, image):
        self.image = image         # OpenCV image array
        self.drawn = 0          # Count of how many detector-boxes have been drawn
        self.path = "/usr/local/lib/python2.7/dist-packages/SimpleCV/Features/HaarCascades/"      # The path to the haarcascades data xml files
        self.rects = []         # Discovered rectangles from Image Analysis
 
    def detect(self, xml):          # Detect people in image and save the image bounds around them
        cascade = cv2.CascadeClassifier(self.path + xml)
        #self.image = cv2.imread(self.image_name)  # Loads the image into a numpy array
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hits = cascade.detectMultiScale(      # Grayscale and analyze image using 
            gray,                 #  the selected cascade classifier file
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        self.rects.append(hits)     # Add detected people to rect-list for drawing
        return hits         # Can use len(hits) to check if anyone was found
 
    # The following functions provide an xml file to be used as the cascade classifier
    #  to detect different things, such as face, upper body, or pedestrian
    def face(self):
        return self.detect('face.xml')
 
    def face2(self):
        return self.detect('face2.xml')
 
    def face3(self):
        return self.detect('face3.xml')
 
    def full_body(self):
        return self.detect('haarcascade_fullbody.xml')
 
    def upper_body(self):
        return self.detect('haarcascade_upperbody.xml')
 
    def pedestrian(self):
        return self.detect("hogcascade_pedestrians.xml")
 
    # This function will draw the rectangles around all objects found and then 
    #  overwrite the original image file.
    def draw(self):
        for hits in self.rects:
            color = (0, 0, 255)# Rect color selection
            self.drawn += 1
            for (x,y,w,h) in hits:
                cv2.rectangle(self.image, (x, y), (x+w, y+h), color, 2) # Draws the Rect
        #cv2.imwrite(self.image_name, self.image)    # Saves the file over the original image name
        return hits
    def run(self):
        faces = len(self.face())
        #faces += len(self.face2())
        #faces += len(self.face3())
        #uppers = len(self.upper_body())
        #fulls = len(self.full_body())
        #peds = len(self.pedestrian())
        self.draw()
        #self.overlay()
   
        return faces, self.rects
