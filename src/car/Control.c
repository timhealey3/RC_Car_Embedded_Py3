#include <stdio.h>
#include "Control.h"
#include <pigpio.h>

void GPIOSetup() {
    // set up outputs
    gpioSetMode(PIN1_MOTOR1, PI_OUTPUT);
    gpioSetMode(PIN2_MOTOR1, PI_OUTPUT);
    gpioSetMode(PIN1_MOTOR2, PI_OUTPUT);
    gpioSetMode(PIN2_MOTOR2, PI_OUTPUT);
    gpioSetMode(PIN1_MOTOR3, PI_OUTPUT);
    gpioSetMode(PIN2_MOTOR3, PI_OUTPUT);
    gpioSetMode(PIN1_MOTOR4, PI_OUTPUT);
    gpioSetMode(PIN2_MOTOR4, PI_OUTPUT);
    // set status
    gpioWrite(PIN1_MOTOR1, PI_LOW);
    gpioWrite(PIN1_MOTOR1, PI_LOW);
    gpioWrite(PIN1_MOTOR2, PI_LOW);
    gpioWrite(PIN1_MOTOR2, PI_LOW);
    gpioWrite(PIN1_MOTOR3, PI_LOW);
    gpioWrite(PIN1_MOTOR3, PI_LOW);
    gpioWrite(PIN1_MOTOR4, PI_LOW);
    gpioWrite(PIN1_MOTOR4, PI_LOW);
}

void pwmDutyCycle(int cycleNum) {
    gpioPWM(PIN_1A, cycleNum);
    gpioPWM(PIN_1B, cycleNum);
    gpioPWM(PIN_2A, cycleNum);
    gpioPWM(PIN_2B, cycleNum);
}

void motorsStop() {
    pwmDutyCycle(0);
    gpioWrite(PIN1_MOTOR1, PI_LOW);
    gpioWrite(PIN2_MOTOR1, PI_LOW);
    gpioWrite(PIN1_MOTOR2, PI_LOW);
    gpioWrite(PIN2_MOTOR2, PI_LOW);
    gpioWrite(PIN1_MOTOR3, PI_LOW);
    gpioWrite(PIN2_MOTOR3, PI_LOW);
    gpioWrite(PIN1_MOTOR4, PI_LOW);
    gpioWrite(PIN2_MOTOR4, PI_LOW);
}

void cleanup() {
    // stop motors
    motorsStop();    
    // clean gpio
    gpioTerminate();
}

void pwmFrequency(int freqNum) {
    gpioSetPWMfrequency(PIN_1A, freqNum);
    gpioSetPWMfrequency(PIN_1B, freqNum);
    gpioSetPWMfrequency(PIN_2A, freqNum);
    gpioSetPWMfrequency(PIN_2B, freqNum);
}

void PWMSetup() {
    gpioSetMode(PIN_1A, PI_OUTPUT);
    gpioSetMode(PIN_1B, PI_OUTPUT);
    gpioSetMode(PIN_2A, PI_OUTPUT);
    gpioSetMode(PIN_2B, PI_OUTPUT);
}

int convertThrottle(int throttle) {
    int convertedThrottle;
    if (throttle == 0) {
    	convertedThrottle = 0;
    } else if (throttle == 1) {
    	convertedThrottle = 128;
    } else if (throttle == 2) {
	convertedThrottle = 192;
    } else if (throttle == 3) {
	convertedThrottle = 224;	    
    } else {
	convertedThrottle = 255;
    }
    return convertedThrottle;
}

void motorsForward(int throttle) {
    int convertedThrottle = convertThrottle(throttle);
    pwmDutyCycle(convertedThrottle);
    gpioWrite(PIN1_MOTOR1, PI_HIGH);
    gpioWrite(PIN2_MOTOR1, PI_LOW);
    gpioWrite(PIN1_MOTOR2, PI_HIGH);
    gpioWrite(PIN2_MOTOR2, PI_LOW);
    gpioWrite(PIN1_MOTOR3, PI_HIGH);
    gpioWrite(PIN2_MOTOR3, PI_LOW);
    gpioWrite(PIN1_MOTOR4, PI_HIGH);
    gpioWrite(PIN2_MOTOR4, PI_LOW);
}

void turnRight() {
    // turn motor 2 off
    gpioWrite(PIN1_MOTOR2, PI_LOW);
    gpioWrite(PIN2_MOTOR2, PI_LOW);
}

void turnLeft() {
    // turn motor 1
    gpioWrite(PIN1_MOTOR1, PI_LOW);
    gpioWrite(PIN2_MOTOR1, PI_LOW);
}

int setup() {
    printf("Set up Car process start\n");
    // init pigpio
    if (gpioInitialise() < 0) {
        printf("pigpio initialization failed.\n");
        return 1;
    }

    GPIOSetup();
    PWMSetup();
    pwmFrequency(100);
    pwmDutyCycle(0);
    return 0;
}
