import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.IN)         #Read output from PIR motion sensor

             
def Occupied():
    i = GPIO.input(11)
    if i==0:
        return False
    elif i == 1:
        return True
if __name__ == '__main__':
    while True:
        print Occupied()
        time.sleep(0.1)
