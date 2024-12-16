import RPi.GPIO as GPIO
import time

import RPi.GPIO as GPIO


try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.OUT)
    GPIO.setup(27, GPIO.OUT)
    GPIO.output(17, GPIO.HIGH)
    GPIO.output(27, GPIO.LOW)
    time.sleep(2)
    GPIO.output(17, GPIO.LOW)
    GPIO.output(27, GPIO.LOW)
finally:
    # Clean up GPIO settings
    GPIO.cleanup()

