import sys
from logging import Logger
import Control
import Telemetry
sys.path.append('/home/timh/codingProjects/src/camera')
import Camera

class RC_Car:
    def __init__(self, speed = 0, direction = 0):
        self.speed = speed
        self.direction = direction
        self.autonomous_mode = False
        self.training_mode = False
        self.controls = Control()
        self.telemetry = Telemetry()
        self.camera = Camera()

    def take_photo(self):
        forward_data, left_data, right_data = self.telemetry.get_training_data()
        self.camera.take_photo_training(forward_data, left_data, right_data)

    def turn(self, newDirection):
        Logger.debug("Turning car")
        self.direction += newDirection
        Logger.debug(f"New direction: {self.direction}")
        if newDirection == -1:
            self.turn_right()
        elif newDirection == 1:
            self.turn_left()
        else:
            Logger.warning("No direction specified in turn function")
        
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
