import cv2
import cv2.cv as cv
import numpy as np
from WebcamStream import WebcamVideoStream
def FaceTracking(curFrame, lastFrame, faces):
    hsv = cv2.cvtColor(curFrame, cv.CV_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    trackFaces = []
    for face in faces:
        print 'Track'
        x0, y0, w, h = face
        x0 = int(x0)
        y0= int(y0)
        w = int(w)
        h = int(h)
        x1 = x0 + w -1
        y1 = y0 + h -1
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        hist_flat = hist.reshape(-1)
        prob = cv2.calcBackProject([hsv,cv2.cvtColor(lastFrame, cv.CV_BGR2HSV)], [0], hist_flat, [0, 180], 1)
        prob &= mask
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        new_ellipse, (x2,y2,w2,h2)= cv2.CamShift(prob, (x0,y0,w,h), term_crit)
        trackFace = (int(x2), int(y2), int(w2), int(h2))
        trackFaces.append(trackFace)
    return trackFaces 
if __name__ == '__main__':
    stream = WebcamVideoStream(src=0).start()
    (x,y,w,h)= (125,125,200,100) # get bounding box from some method
    faces = []
    faces.append((x,y,w,h))
    color = (0,0,255)
    img = stream.read()
    while True:
        try:
            img1 = stream.read()
            faces = FaceTracking(img1, img, faces)
            (x, y, w, h) = faces[0]
            pt1 = (int(x), int(y))
            pt2 = (int((x+w)), int(y + h))
            img = img1
            #draw bounding box on img1
            cv2.rectangle(img1,pt1, pt2, color, 2) 
            cv2.imshow("CAMShift",img1)
        except KeyboardInterrupt:
            stream.stop()
            break
    
