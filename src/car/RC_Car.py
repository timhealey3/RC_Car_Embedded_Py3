from logging import Logger

import Control

class RC_Car:
    def __init__(self, speed = 0, direction = 0):
        self.speed = speed
        self.direction = direction
        self.autonomous_mode = False
        self.training_mode = False
        self.controls = Control()

    def turn(self, newDirection):
        Logger.debug("Turning car")
        self.direction += newDirection
        Logger.debug(f"New direction: {self.direction}")
        match newDirection:
            case -1:
                self.turn_right()
            case 1:
                self.turn_left()
            case _:
                Logger.warning("No Direction specified in turn statement")


    def turn_left(self):
        Logger.debug("Turning car left")
        self.controls.turnLeft()

    def turn_right(self):
        Logger.debug("Turning car right")
        self.controls.turnRight()

    def accelerate(self, newSpeed):
        print("Accelerating car")
        self.speed += newSpeed
        print(f"New speed: {self.speed}")
        self.controls.motorsForward()

    def telemetry(self):
        print(f"Speed: {self.speed}")
        print(f"Direction: {self.direction}")
        return self.speed, self.direction
