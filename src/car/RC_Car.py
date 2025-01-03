import sys
from logging import Logger
from Control import Control
from Telemetry import Telemetry
sys.path.append('/home/timh/codingProjects/src/camera')
# from Camera import Camera

class RC_Car:
    def __init__(self):
        self.speed = 0
        self.direction = 0
        self.autonomous_mode = False
        self.training_mode = False
        self.controls = Control()
        self.telemetry = Telemetry()
        # self.camera = Camera()

    def take_photo(self):
        forward_data, left_data, right_data = self.telemetry.get_training_data()
        # self.camera.take_photo_training(forward_data, left_data, right_data)

    def turn(self, newDirection):
        self.direction += newDirection
        if newDirection == -1:
            self.turn_right()
        elif newDirection == 1:
            self.turn_left()
        else:
            Logger.warning("No direction specified in turn function")
        
    def turn_left(self):
        self.controls.turnLeft()

    def turn_right(self):
        self.controls.turnRight()

    def stop(self):
        print("stop car")
        self.controls.motorsStop()

    def turnOff(self):
        print("turning off car")
        self.controls.cleanUp()
    
    def accelerate(self):
        print("Accelerating car")
        self.controls.motorsForward()
