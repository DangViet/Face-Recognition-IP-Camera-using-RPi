import cv2
import Image
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import StringIO
import time

#from imutils.video import WebcamVideoStream
#from imutils.video import FPS
#import imutils
#from imutils.video.pivideostream import PiVideoStream
from CameraStream import CameraStream
from WebcamStream import WebcamVideoStream
from picamera.array import PiRGBArray
from picamera import PiCamera

#from FaceDetection import FaceDetection
from ProcImg import ProcImg
#from NewProcImg import ProcImg
from PIR import Occupied
count = 0
class CamHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_respond(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
    def do_AUTHHEAD(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm=\"Test\"')
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                if Occupied():
                    try:
                        '''
                        img = capture.read()
                        #insert image processing function here
                        FaceDetector = FaceDetection(img)
                        numFace, faceRect = FaceDetector.run()
                        '''
                        ProcessImage.threadedProcess()
                        if not ProcessImage.isEmpty():
                            img = ProcessImage.getFrame()
                            # Create MJpeg Stream
                            imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            jpg = Image.fromarray(imgRGB)
                            tmpFile = StringIO.StringIO()
                            jpg.save(tmpFile,'JPEG')
                            self.wfile.write("--jpgboundary")
                            self.send_header('Content-type','image/jpeg')
                            self.send_header('Content-length',str(tmpFile.len))
                            self.end_headers()
                            jpg.save(self.wfile,'JPEG')
                    
                    except KeyboardInterrupt:
                        break
                        return
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="http://127.0.0.1:8080/cam.mjpg"/>')
            self.wfile.write('</body></html>')
            return
def main():
    
    stream = WebcamVideoStream(src=0).start() 
    global ProcessImage 
    ProcessImage = ProcImg(stream)
    try:
        server = HTTPServer(('',8080),CamHandler)
        print "server started"
        server.serve_forever()
    except KeyboardInterrupt:
        stream.stop()
        cv2.destroyAllWindows()
        server.socket.close()

if __name__ == '__main__':
    main()
