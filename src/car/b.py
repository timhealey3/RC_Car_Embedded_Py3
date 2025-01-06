import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
PIN = 17  # Example pin
GPIO.setup(PIN, GPIO.OUT)

# Blink the pin
try:
    while True:
        GPIO.output(PIN, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(PIN, GPIO.LOW)
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting test...")
finally:
    GPIO.cleanup()

