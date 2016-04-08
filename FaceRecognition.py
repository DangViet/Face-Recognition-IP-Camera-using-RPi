import cv2

import config
model = cv2.createLBPHFaceRecognizer()
#model = cv2.createFisherFaceRecognizer()
model.load(config.TRAINING_FILE)
clahe = cv2.createCLAHE()
def resize(image):
    """Resize a face image to the proper size for training and detection.
    """
    temp = cv2.resize(image, (config.FACE_WIDTH, config.FACE_HEIGHT),interpolation=cv2.INTER_LANCZOS4)
    return temp

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

def faceRecognition(image, rects):
    labels = []
    conf = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for (x, y, w, h) in rects:
        cropped = crop(gray, x, y, w, h)
        resized = resize(cropped)
        #equ = cv2.equalizeHist(resized)
        equ = clahe.apply(resized)
        label, confidence = model.predict(equ)
        labels.append(label)
        conf.append(confidence)
    return labels, conf    
