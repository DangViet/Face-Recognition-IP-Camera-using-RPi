import sys, math, Image
import os
import cv2
import numpy as np
import config
import copy
import cv2.cv as cv
def Distance(p1,p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_h
    # scale factor
    scale = float(dist)/float(reference)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
    image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
    return image
def crop(image, x, y, w, h):
    """Crop box defined by x, y (upper left corner) and w, h (width and height)
    to an image with the same aspect ratio as the face training data.  Might
    return a smaller crop if the box is near the edge of the image.
    """
    crop_height = int((config.FACE_HEIGHT / float(config.FACE_WIDTH)) * w)
    midy = y + h/2
    y1 = max(0, midy-crop_height/2)
    y2 = min(image.shape[0]-1, midy+crop_height/2)
    return copy.copy(image[y1:y2, x:x+w])

def calcHist(image, rects):
    allRoiHist = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in rects:
        cropped = crop(gray, x, y, w, h)
        roiImg= crop(image, x, y, w, h)    
        eye_cascade = cv2.CascadeClassifier('/home/pi/HaarCascades/haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(cropped)
    
        if len(eyes) == 2 and cropped is not None:
            (ex1,ey1,ew1,eh1) = eyes[0]
            (ex2,ey2,ew2,eh2) = eyes[1]
            leftEye = (ex1+ ew1/2, ey1+eh1/2)
            rightEye = (ex2+ ew2/2, ey2+eh2/2)
            if ex1 > ex2:
                leftEye, rightEye = rightEye, leftEye
            

            alignFace =  CropFace(Image.fromarray(roiImg), eye_left=leftEye,eye_right=rightEye)
            cv2Img = np.asarray(alignFace)
            roi = cv2.cvtColor(cv2Img, cv2.COLOR_BGR2HSV)
            roiHist = cv2.calcHist([roi], [0], None, [16], [0,180])
            roiHist = cv2.normalize(roiHist, roiHist, 0,255, cv2.NORM_MINMAX)
            allRoiHist.append(roiHist)
    return allRoiHist

if __name__=='__main__':
    for folder in next(os.walk(path))[1]:
        for file in os.listdir(path+'/'+folder):
            image_path = path + '/' + folder + '/' + file
            image = Image.open(image_path)
            array = np.array(image, 'uint8')
            eyes = eye_cascade.detectMultiScale(array)

            if len(eyes) == 2:
                (ex1,ey1,ew1,eh1) = eyes[0]    
                (ex2,ey2,ew2,eh2) = eyes[1]
                leftEye = (ex1+ ew1/2, ey1+eh1/2)
                rightEye = (ex2+ ew2/2, ey2+eh2/2)
                if ex1 > ex2:
                    leftEye, rightEye = rightEye, leftEye
                
                image = CropFace(image, eye_left=leftEye,eye_right=rightEye)
                #cv2.imwrite('./AlignFace/'+file,image, [cv2.IMWRITE_PNG_COMPRESSION, 0] )  
                image.save('./AlignFace/'+file)
              
         
            
