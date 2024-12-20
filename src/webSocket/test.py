import RPi.GPIO as GPIO
import time

# Set up GPIO mode and pin
GPIO.setmode(GPIO.BCM)      # Use Broadcom pin-numbering
GPIO.setup(19, GPIO.OUT)    # Set GPIO 18 as output

# Set up PWM on GPIO 18 with a frequency of 2 kHz
pwm = GPIO.PWM(19, 2000)    # 2 kHz frequency (2000 Hz)

# Start PWM with a duty cycle of 0% (off initially)
pwm.start(0)

try:
    while True:
        # Gradually increase the duty cycle from 0 to 100%
        for dc in range(0, 101, 5):
            pwm.ChangeDutyCycle(dc)  # Change PWM duty cycle
            print(f"Duty Cycle: {dc}%")
            time.sleep(0.1)  # Delay for 0.1 seconds
            
        # Gradually decrease the duty cycle from 100% to 0%
        for dc in range(100, -1, -5):
            pwm.ChangeDutyCycle(d)
            print(f"Duty Cycle: {dc}%")
            time.sleep(0.1)

except KeyboardInterrupt:
    # If the program is interrupted (Ctrl+C), stop the PWM
    print("PWM stopped")
    pwm.stop()

finally:
    # Clean up GPIO settings
    GPIO.cleanup()
c
