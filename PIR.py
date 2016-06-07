import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.IN)         #Read output from PIR motion sensor

             
def Occupied():
    if not hasattr(Occupied, "count"):
        Occupied.count = 0
    i = GPIO.input(11)
    if i==0:
        if Occupied.count < 20:
            #Occupied.count += 1
            return True
        else:
            return False
    elif i == 1:
        Occupied.count = 0
        return True
if __name__ == '__main__':
    while True:
        print Occupied()
        time.sleep(0.1)
