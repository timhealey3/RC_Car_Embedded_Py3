import RPi.GPIO as GPIO
import time

# This class controls the two motors on the car

class Control:
    def __init__(self):
        # set up gpio output pins for motors one and two
        GPIO.setmode(GPIO.BCM)
        self.IN1_MOTOR1_PIN = 17
        self.IN2_MOTOR1_PIN = 27   
        self.IN1_MOTOR2_PIN = 22
        self.IN2_MOTOR2_PIN = 23

        GPIO.setup(self.IN1_MOTOR1_PIN, GPIO.OUT)
        GPIO.setup(self.IN2_MOTOR1_PIN, GPIO.OUT)             
        GPIO.setup(self.IN1_MOTOR2_PIN, GPIO.OUT)
        GPIO.setup(self.IN2_MOTOR2_PIN, GPIO.OUT) 

        # pwm set up
        # GPIO.setup(18, GPIO.OUT)   
        # self.pwm = GPIO.PWM(18, 100)
        # self.pwm.start(0)
    
    def motorsForward(self):
        # move motor one forward
        GPIO.output(self.IN1_MOTOR1_PIN, GPIO.HIGH)
        GPIO.output(self.IN2_MOTOR1_PIN, GPIO.LOW)
        # move motor two forward
        GPIO.output(self.IN1_MOTOR2_PIN, GPIO.HIGH)
        GPIO.output(self.IN2_MOTOR2_PIN, GPIO.LOW)

    def turnLeft(self):
        # turn motor 1 off
        GPIO.output(self.IN1_MOTOR1_PIN, GPIO.LOW)
        GPIO.output(self.IN2_MOTOR1_PIN, GPIO.LOW)

    def turnRight(self):
        # turn motor 2 off
        GPIO.output(self.IN1_MOTOR2_PIN, GPIO.LOW)
        GPIO.output(self.IN2_MOTOR2_PIN, GPIO.LOw)

    def motorsStop(self):
        # stop motor one
        GPIO.output(self.IN1_MOTOR1_PIN, GPIO.LOW)
        GPIO.output(self.IN2_MOTOR1_PIN, GPIO.LOW)
        # stop motor two
        GPIO.output(self.IN1_MOTOR2_PIN, GPIO.LOW)
        GPIO.output(self.IN2_MOTOR2_PIN, GPIO.LOW)

    def cleanUp(self):
        # Clean up GPIO settings
        GPIO.cleanup()
        # self.pwm.stop()
