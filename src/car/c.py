import RPi.GPIO as GPIO

class Control:
    def __init__(self):
        print("Initializing GPIO...")
        GPIO.setmode(GPIO.BCM)
        self.IN1_MOTOR1_PIN = 17
        self.IN2_MOTOR1_PIN = 27
        GPIO.setup(self.IN1_MOTOR1_PIN, GPIO.OUT)
        GPIO.setup(self.IN2_MOTOR1_PIN, GPIO.OUT)
        print("GPIO initialized")

    def motorsForward(self):
        print("Moving motor forward...")
        GPIO.output(self.IN1_MOTOR1_PIN, GPIO.HIGH)
        GPIO.output(self.IN2_MOTOR1_PIN, GPIO.LOW)
        print("Motor forward signal sent")

control = Control()
control.motorsForward()

