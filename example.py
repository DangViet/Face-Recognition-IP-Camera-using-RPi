import cv2
import cv2.cv as cv
import numpy as np
import config
from FaceDetection import FaceDetection
import sys, os, select, copy
from PIL import Image
import time
camera = cv2.VideoCapture(0)
camera.set(3,640)
camera.set(4,480)
time.sleep(0.5)
def resize(image):
    """Resize a face image to the proper size for training and detection.
    """
    temp = cv2.resize(image, (config.FACE_WIDTH, config.FACE_HEIGHT),interpolation=cv2.INTER_LANCZOS4)
    return temp
def captureImage():
    grab, img = camera.read()
    return img
def crop(image, x, y, w, h):
    """Crop box defined by x, y (upper left corner) and w, h (width and height)
    to an image with the same aspect ratio as the face training data.  Might
    return a smaller crop if the box is near the edge of the image.
    """
    crop_height = int((config.FACE_HEIGHT / float(config.FACE_WIDTH)) * w)
    midy = y + h/2
    y1 = max(0, midy-crop_height/2)
    y2 = min(image.shape[0]-1, midy+crop_height/2)
    return image[y1:y2, x:x+w]
if __name__ == '__main__':
    path = './temp/'
    numPic = 1
    i = 0
    while i < numPic:
        image = captureImage()

        scale = cv2.resize(image, (0,0), fx = 1/float(2), fy = 1/float(2))

        gray = cv2.cvtColor(scale, cv2.COLOR_RGB2GRAY)

        equal = cv2.equalizeHist(gray)

        detector = FaceDetection(image)
        numFace, rect = detector.run()
        if numFace != 1:
            print 'Cannot detect a single face'
            continue

        #cv2.imwrite(path+str(i)+'original.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        #detector.draw()
        #time.sleep(0.5)
        #cv2.imwrite(path+str(i)+'scale.png', scale, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        #cv2.imwrite(path+str(i)+'gray.png', gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #cv2.imwrite(path+str(i)+'equal.png', equal, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        (x, y, w, h) = rect[0]
        cropped = crop(image, x, y, w, h)

        cv2.imwrite(path+'mark.png', cropped, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        i+=1
    #cv2.destroyAllWindows()
