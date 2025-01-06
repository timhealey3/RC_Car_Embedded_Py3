import RPi.GPIO as GPIO
import pigpio
# This class controls the two motors on the car

class Control:
    def __init__(self):
        # set up PWM pigpio pins
        self.pi = pigpio.pi()
        self.pin1A = 12
        self.pin1B = 16
        self.pin2A = 22
        self.pin2B = 25
        # set up gpio output pins for motors one and two
        GPIO.setmode(GPIO.BCM)
        # front motors
        self.IN1_MOTOR1_PIN = 17
        self.IN2_MOTOR1_PIN = 27   
        self.IN1_MOTOR2_PIN = 23
        self.IN2_MOTOR2_PIN = 24
        # back motors
        self.IN1_MOTOR3_PIN = 26
        self.IN2_MOTOR3_PIN = 13
        self.IN1_MOTOR4_PIN = 6
        self.IN2_MOTOR4_PIN = 5
        # set up
        GPIO.setup(self.IN1_MOTOR1_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2_MOTOR1_PIN, GPIO.OUT, initial=GPIO.LOW)             
        GPIO.setup(self.IN1_MOTOR2_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2_MOTOR2_PIN, GPIO.OUT, initial=GPIO.LOW) 
        GPIO.setup(self.IN1_MOTOR3_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2_MOTOR3_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN1_MOTOR4_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2_MOTOR4_PIN, GPIO.OUT, initial=GPIO.LOW)
        self.pi.set_PWM_frequency(16, 100)       # 100Hz frequency
        self.pi.set_PWM_dutycycle(16, 128)       # 50% duty cycle (128 out of 255)
        self.pi.set_PWM_frequency(12, 100)       # 100Hz frequency
        self.pi.set_PWM_dutycycle(12, 128)       # 50% duty cycle (128 out of 255)

        self.pi.set_PWM_frequency(22, 100)       # 100Hz frequency
        self.pi.set_PWM_dutycycle(22, 128)       # 50% duty cycle (128 out of 255)
        self.pi.set_PWM_frequency(25, 100)       # 100Hz frequency
        self.pi.set_PWM_dutycycle(25, 128)       # 50% duty cycle (128 out of 255)

    def motorsForward(self):
        # move motor one forward
        GPIO.output(self.IN1_MOTOR1_PIN, GPIO.HIGH)
        GPIO.output(self.IN2_MOTOR1_PIN, GPIO.LOW)
        # move motor two forward
        GPIO.output(self.IN1_MOTOR2_PIN, GPIO.HIGH)
        GPIO.output(self.IN2_MOTOR2_PIN, GPIO.LOW)
        # move motor three forward
        GPIO.output(self.IN1_MOTOR3_PIN, GPIO.LOW)
        GPIO.output(self.IN2_MOTOR3_PIN, GPIO.HIGH)
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
        self.pi.set_PWM_dutycycle(12, 0)
        self.pi.set_PWM_dutycycle(16, 0)
        self.pi.set_PWM_dutycycle(22, 0)
        self.pi.set_PWM_dutycycle(25, 0)
        self.pi.stop()
        # self.pwm.stop() 
