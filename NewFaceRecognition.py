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

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx,ny = x,y = center
    sx=sy=1.0
    if new_center:
        (nx,ny) = new_center
    if scale:
        (sx,sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)
#def Rotate():
    
    
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
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
    image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image
cascadePath = '/home/pi/HaarCascades/haarcascade_eye.xml'
path = './samples'
model = cv2.createLBPHFaceRecognizer()
model.load(config.TRAINING_FILE_LBPH)
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

def NewFaceRecognition(image, rects):
    labelAndConf = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in rects:
        cropped = crop(gray, x, y, w, h)
        
        eye_cascade = cv2.CascadeClassifier(cascadePath)
        eyes = eye_cascade.detectMultiScale(cropped)
    
        if len(eyes) == 2 and cropped is not None:
            (ex1,ey1,ew1,eh1) = eyes[0]
            (ex2,ey2,ew2,eh2) = eyes[1]
            leftEye = (ex1+ ew1/2, ey1+eh1/2)
            rightEye = (ex2+ ew2/2, ey2+eh2/2)
            if ex1 > ex2:
                leftEye, rightEye = rightEye, leftEye
            

            alignFace =  CropFace(Image.fromarray(cropped), eye_left=leftEye,eye_right=rightEye)
            cv2Img = np.asarray(alignFace)
            equalHist= cv2.equalizeHist(cv2Img) 
            label, confidence = model.predict(equalHist)
            labelAndConf.append((label, confidence))
        else: 
            labelAndConf.append((0, 0))
    return labelAndConf

if __name__=='__main__':
    eye_cascade = cv2.CascadeClassifier(cascadePath)
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
                image.save('./AlignFace/'+folder+'_'+file)
              
         
            
