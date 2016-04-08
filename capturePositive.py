import cv2
import os
from PIL import Image
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import config
import time
from FaceDetection import FaceDetection
import sys, os, glob, select, copy
import config
# Prefix for positive training image filenames.
POSITIVE_FILE_PREFIX = 'positive_'
camera = cv2.VideoCapture(0)
#rawCapture = PiRGBArray(camera)
time.sleep(0.1)
def resize(image):
    """Resize a face image to the proper size for training and detection.
    """
    temp = cv2.resize(image, (config.FACE_WIDTH, config.FACE_HEIGHT),interpolation=cv2.INTER_LANCZOS4)
    return temp
def captureImage():
    grab, img = camera.read()
    #cv2.imshow('Preview',img)
    #cv2.waitKey(1000)
    #cv2.destroyAllWindows()
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
def is_letter_input(letter):
    # Utility function to check if a specific character is available on stdin.
    # Comparison is case insensitive.
    if select.select([sys.stdin,],[],[],0.0)[0]:
        input_char = sys.stdin.read(1)
        if input_char != '':
            print input_char
        return input_char.lower() == letter.lower()
    return False
def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    #image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    clahe = cv2.createCLAHE()
    for folder in next(os.walk(path))[1]:
        for file in os.listdir(path+'/'+folder):
            image_path = path+'/'+folder+'/'+file
            # Read the image and convert to grayscale
            image_pil = Image.open(image_path).convert('L')
            #image = cv2.imread(image_path)

            #image_pil = cv2.cvtColor(image_pil, cv2.COLOR_BGR2GRAY)
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            # Get the label of the image
            nbr = int(folder.replace("subject", ""))
            print(nbr)
            '''
            # Detect the face in the image
            # For face detection we will use the Haar Cascade provided by OpenCV.
            cascadePath = "/home/pi/HaarCascades/lbpcascade_frontalface.xml"
            faceCascade = cv2.CascadeClassifier(cascadePath)
            # Detect the face in the image
            faces = faceCascade.detectMultiScale(image)
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                cropped = crop(image, x, y, w, h)
                resized = resize(cropped)
                equHist = cv2.equalizeHist(resized)
                #equHist = clahe.apply(resized)
                images.append(equHist)
                labels.append(nbr)
            '''
            image = clahe.apply(image)
            images.append(image)
            labels.append(nbr)
            
            cv2.imshow("Adding faces to traning set...", image)
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels
if __name__ == '__main__':
    # Create the directory for positive training images if it doesn't exist.
    #if not os.path.exists(config.POSITIVE_PATH):
    #    os.makedirs(config.POSITIVE_PATH)
    # Find the largest ID of existing positive images.
    # Start new images after this ID value.
    #files = sorted(glob.glob(os.path.join(config.POSITIVE_PATH, 
    #    POSITIVE_FILE_PREFIX + '[0-9][0-9][0-9].pgm')))
    #count = 0
    #if len(files) > 0:
        # Grab the count from the last filename.
        #count = int(files[-1][-7:-4])+1
    #model = cv2.createFisherFaceRecognizer()
    #model.load(config.TRAINING_FILE)
    # Path to the Yale Dataset
    path = './samples'
    # Call the get_images_and_labels function and get the face images and the 
    # corresponding labels
    images, labels = get_images_and_labels(path)
    cv2.destroyAllWindows()
    # For face recognition we will the the LBPH Face Recognizer 
    model= cv2.createLBPHFaceRecognizer()
    model.train(images, np.array(labels))
    model.save(config.TRAINING_FILE)
    print 'Training data saved to', config.TRAINING_FILE
    '''
    print 'Capturing positive training images.'
    print 'Press button or type c (and press enter) to capture an image.'
    print 'Press Ctrl-C to quit.'
    while True:
        # Check if button was pressed or 'c' was received, then capture image.
        if is_letter_input('c'):
            print 'Capturing image...'
            image = captureImage()
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            detector = FaceDetection(image)
            numFace, rect = detector.run()
            print rect
            if numFace != 1:
                print 'Could not detect single face!  Check the image in capture.pgm' \
                      ' to see what was captured and try again with only one face visible.'
                continue
            (x, y, w, h) = rect[0]
            # Crop image as close as possible to desired face aspect ratio.
            # Might be smaller if face is near edge of image.
            cropped = crop(gray, x, y, w, h)
            cropped = resize(cropped)
            #cv2.equalizeHist(cropped, cropped) 
            label, confidence = model.predict(cropped)
            print label
            print 'Predicted {0} face with confidence {1} (lower is more confident).'.format('POSITIVE' if label == config.POSITIVE_LABEL else 'NEGATIVE', confidence)
            if label == config.POSITIVE_LABEL and confidence < config.POSITIVE_THRESHOLD:
                print 'Recognized face!'
            else:
                print 'Did not recognize face!'
            # Save image to file.
            #filename = os.path.join(config.POSITIVE_PATH, POSITIVE_FILE_PREFIX + '%03d.pgm' % count)
            #cv2.imwrite(filename, crop)
            #print 'Found face and wrote training image', filename
            #count += 1
       '''     
