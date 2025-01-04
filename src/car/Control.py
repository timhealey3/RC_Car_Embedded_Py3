import RPi.GPIO as GPIO

# This class controls the two motors on the car
class Control:
    def __init__(self):
        # set up gpio output pins for motors one and two
        GPIO.setmode(GPIO.BCM)
        # front motors
        self.IN1_MOTOR1_PIN = 17
        self.IN2_MOTOR1_PIN = 27   
        self.IN1_MOTOR2_PIN = 23
        self.IN2_MOTOR2_PIN = 24
        self.ENA_FRONT = 18
        self.ENB_FRONT = 19
        # back motors
        self.IN1_MOTOR3_PIN = 26
        self.IN2_MOTOR3_PIN = 16
        self.IN1_MOTOR4_PIN = 6
        self.IN2_MOTOR4_PIN = 5
        self.ENA_BACK = 12
        self.ENB_BACK = 13

        # set up front motors
        GPIO.setup(self.IN1_MOTOR1_PIN, GPIO.OUT)
        GPIO.setup(self.IN2_MOTOR1_PIN, GPIO.OUT)             
        GPIO.setup(self.IN1_MOTOR2_PIN, GPIO.OUT)
        GPIO.setup(self.IN2_MOTOR2_PIN, GPIO.OUT)
        # set up back motors
        GPIO.setup(self.IN1_MOTOR3_PIN, GPIO.OUT)
        GPIO.setup(self.IN2_MOTOR3_PIN, GPIO.OUT)
        GPIO.setup(self.IN1_MOTOR4_PIN, GPIO.OUT)
        GPIO.setup(self.IN2_MOTOR4_PIN, GPIO.OUT)
        # pwm set up
        GPIO.setup(self.ENA_FRONT, GPIO.OUT)
        GPIO.setup(self.ENB_FRONT, GPIO.OUT)
        GPIO.setup(self.ENA_BACK, GPIO.OUT)
        GPIO.setup(self.ENB_BACK, GPIO.OUT)
        # set pwm freq
        self.PWM_ENA_FRONT = GPIO.PWM(self.ENA_FRONT, 100)
        self.PWM_ENB_FRONT = GPIO.PWM(self.ENB_FRONT, 100)
        self.PWM_ENA_BACK = GPIO.PWM(self.ENA_BACK, 100)
        self.PWM_ENB_BACK = GPIO.PWM(self.ENB_BACK, 100)
        # start PWM
        self.PWM_ENA_FRONT.start(0)
        self.PWM_ENB_FRONT.start(0)
        self.PWM_ENA_BACK.start(0)
        self.PWM_ENB_BACK.start(0)
    
    def motorsForward(self):
        self.PWM_ENA_FRONT.ChangeDutyCycle(25)
        self.PWM_ENB_FRONT.ChangeDutyCycle(25)
        self.PWM_ENA_BACK.ChangeDutyCycle(25)
        self.PWM_ENB_BACK.ChangeDutyCycle(25)
        # move motor one forward
        GPIO.output(self.IN1_MOTOR1_PIN, GPIO.HIGH)
        GPIO.output(self.IN2_MOTOR1_PIN, GPIO.LOW)
        # move motor two forward
        GPIO.output(self.IN1_MOTOR2_PIN, GPIO.HIGH)
        GPIO.output(self.IN2_MOTOR2_PIN, GPIO.LOW)
        # move motor three forward
        GPIO.output(self.IN1_MOTOR3_PIN, GPIO.HIGH)
        GPIO.output(self.IN2_MOTOR3_PIN, GPIO.LOW)
        # move motor four forward
        GPIO.output(self.IN1_MOTOR4_PIN, GPIO.HIGH)
        GPIO.output(self.IN2_MOTOR4_PIN, GPIO.LOW)

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
        # stop motor three
        GPIO.output(self.IN1_MOTOR3_PIN, GPIO.LOW)
        GPIO.output(self.IN2_MOTOR3_PIN, GPIO.LOW)
        # stop motor four
        GPIO.output(self.IN1_MOTOR4_PIN, GPIO.LOW)
        GPIO.output(self.IN2_MOTOR4_PIN, GPIO.LOW)

    def cleanUp(self):
        # Clean up GPIO settings
        GPIO.cleanup()
        # self.pwm.stop()
