import pigpio
import time

pi = pigpio.pi()
motor_pin = 18  # Replace with your actual motor pin

pi.set_PWM_frequency(motor_pin, 100)  # Set frequency to 100Hz

for duty in range(0, 256, 16):  # Test duty cycles in steps of 16
    pi.set_PWM_dutycycle(motor_pin, duty)
    print(f"Duty cycle: {duty}")
    time.sleep(1)  # Wait 1 second to observe motor behavior

pi.set_PWM_dutycycle(motor_pin, 0)  # Turn off motor
pi.stop()

