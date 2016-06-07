import cv2
import cv2.cv as cv
import numpy as np
from WebcamStream import WebcamVideoStream

termination = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
def calcHist(frame, allRoiPts):
    allRoiHist = []
    for (x, y, w, h) in allRoiPts:
        roi = frame[y:y+h, x:x+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_roi =  cv2.inRange(hsv_roi, np.array((0., 40., 80.)), np.array((20., 255., 255.)))
        roiHist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        roiHist = cv2.normalize(roiHist, roiHist, 0,255, cv2.NORM_MINMAX)
        allRoiHist.append(roiHist)
    return allRoiHist
def Track(frame, allRoiPts, allRoiHist):
    hsv = cv2.cvtColor(frame, cv.CV_BGR2HSV)
    trackFaces = []
    tempAllRoiPts = []
    i = 0

    mask = cv2.inRange(hsv, np.array((0., 40.,80.)), np.array((20.,255.,255.)))
    for roiHist in allRoiHist:
        backProj= cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
        backProj&=mask
        (x,y,w,h) = allRoiPts[i]
        if x>0 and y > 0 and w > 0 and h > 0:
            (r, allRoiPts[i])= cv2.CamShift(backProj, (int(x), int(y), int(w), int(h)), termination)
            for j in range(0, 4):
                if allRoiPts[i][j] < 0:
                    allRoiPts[i][j] = 0
                    print 'Lose track'
            pts = np.int0(cv.BoxPoints(r))
            #cv2.polylines(frame, [pts], True, (0, 255,255), 1)         
            i+=1
            trackFaces.append(pts)
    return allRoiPts
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
    
