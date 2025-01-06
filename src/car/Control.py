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
        GPIO.setmode(GPIO.BCM)
        # set motor front pins
        self.IN1_MOTOR1_PIN = 17
        self.IN2_MOTOR1_PIN = 27   
        self.IN1_MOTOR2_PIN = 23
        self.IN2_MOTOR2_PIN = 24
        # set motor back pins
        self.IN1_MOTOR3_PIN = 26
        self.IN2_MOTOR3_PIN = 13
        self.IN1_MOTOR4_PIN = 6
        self.IN2_MOTOR4_PIN = 5
        # set up GPIO
        self.GPIOSetup()
        # pwm set up 100 mhz frequency, duty cycles off
        self.pwmFrequnecy(100)
        self.pwmDutyCycle(0)

    def GPIOSetup(self):
        GPIO.setup(self.IN1_MOTOR1_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2_MOTOR1_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN1_MOTOR2_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2_MOTOR2_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN1_MOTOR3_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2_MOTOR3_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN1_MOTOR4_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2_MOTOR4_PIN, GPIO.OUT, initial=GPIO.LOW)

    def pwmDutyCycle(self, dutyCycleNum):
        self.pi.set_PWM_dutycycle(self.pin1A, dutyCycleNum)
        self.pi.set_PWM_dutycycle(self.pin1B, dutyCycleNum)
        self.pi.set_PWM_dutycycle(self.pin2A, dutyCycleNum)
        self.pi.set_PWM_dutycycle(self.pin2B, dutyCycleNum)

    def pwmFrequnecy(self, freqNum):
        self.pi.set_PWM_frequency(self.pin1A, freqNum)
        self.pi.set_PWM_frequency(self.pin1B, freqNum)
        self.pi.set_PWM_frequency(self.pin2A, freqNum)
        self.pi.set_PWM_frequency(self.pin2B, freqNum)

    def motorsForward(self, pwmThrottle):
        throttle = self.convertThrottle(pwmThrottle)
        self.pwmDutyCycle(throttle)
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

    @staticmethod
    def convertThrottle(self, pwmThrottle):
        return pwmThrottle * 64 if pwmThrottle * 64 < 256 else pwmThrottle * 64 - 1

    def turnLeft(self):
        # turn motor 1 off
        GPIO.output(self.IN1_MOTOR1_PIN, GPIO.LOW)
        GPIO.output(self.IN2_MOTOR1_PIN, GPIO.LOW)

    def turnRight(self):
        # turn motor 2 off
        GPIO.output(self.IN1_MOTOR2_PIN, GPIO.LOW)
        GPIO.output(self.IN2_MOTOR2_PIN, GPIO.LOw)

    def motorsStop(self):
        # stop pwm cycles
        self.pwmDutyCycle(0)
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
        self.motorsStop()
        GPIO.cleanup()
        self.pi.stop()
