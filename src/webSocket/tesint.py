import RPi.GPIO as GPIO
import time

class Car:
    def __init__(self):
        # Set up GPIO pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Motor control pins (IN1, IN2, ENA)
        self.motor_pin1 = 17  # IN1
        self.motor_pin2 = 27  # IN2
        self.ena_pin = 19     # ENA (PWM)

        # Set up motor control pins as output
        GPIO.setup(self.motor_pin1, GPIO.OUT)
        GPIO.setup(self.motor_pin2, GPIO.OUT)
        GPIO.setup(self.ena_pin, GPIO.OUT)

        # Set up PWM on ENA pin
        self.pwm = GPIO.PWM(self.ena_pin, 2000)  # 100 Hz
        self.pwm.start(0)  # Start with 0% duty cycle (motor off)
        print("PWM initialized on ENA pin (GPIO 18)")

    def motorsTesting(self):
        try:
           # Set motor direction (IN1 HIGH, IN2 LOW)
            GPIO.output(self.motor_pin1, GPIO.HIGH)
            GPIO.output(self.motor_pin2, GPIO.LOW)
            print("Motor direction set: IN1 HIGH, IN2 LOW")

            # Run motor at 10% speed using PWM
            print("Running motor at 10% speed...")
            self.pwm.ChangeDutyCycle(10)
            time.sleep(1)

            # Run motor at 50% speed using PWM
            print("Running motor at 50% speed...")
            self.pwm.ChangeDutyCycle(0)
            time.sleep(1)

            # Run motor at 100% speed using PWM
            print("Running motor at 100% speed...")
            self.pwm.ChangeDutyCycle(5)
            time.sleep(1)

            # Run motor at 25% speed using PWM
            print("Running motor at 25% speed...")
            self.pwm.ChangeDutyCycle(2)
            time.sleep(1)

            # Stop the motor
            print("Stopping motor...")
            self.pwm.ChangeDutyCycle(0)
            time.sleep(1)

        finally:
            # Cleanup
            self.pwm.stop()
            GPIO.cleanup()
            print("GPIO cleaned up")

# Instantiate and test
car = Car()
car.motorsTesting()
 
