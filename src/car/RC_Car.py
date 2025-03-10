import ctypes
import threading
import sys
from logging import Logger
from Telemetry import Telemetry
from enum import Enum
sys.path.append('/home/timh/codingProjects/src/camera')
import Camera

class RC_Car:
    def __init__(self):
        print("init car")
        self.speed = 0
        self.direction = 0
        self.right = 0
        self.left = 0
        self.forward = 0
        self.lock = threading.Lock()
        self.autonomous_mode = False
        self.training_mode = False
        self.control = ctypes.CDLL("../car/Control.so")
        self.control.setup()
        self.telemetry = Telemetry()
        self.camera = Camera.Camera()
        print("car set up")

    def training(self):
        if self.training_mode:
            self.take_photo()

    def take_photo(self):
        print("Info: RC_Car - photo taken")
        # forward_data, left_data, right_data = self.telemetry.get_training_data()
        with self.lock:
            self.camera.take_photo_training(self.forward, self.left, self.right)

    def turn(self, newDirection):
        with self.lock:
            self.direction += newDirection
            if newDirection == -1:
                self.turn_right()
            elif newDirection == 1:
                self.turn_left()
            else:
                Logger.warning("No direction specified in turn function")
    
    def update_direction(self, newDirection):
        with self.lock:
            if newDirection == Direction.FORWARD:
                self.forward = 1
            elif newDirection == Direction.BACK:
                self.forward = 0
            elif newDirection == Direction.RIGHT:
                self.left = 0
                self.right = 1
            elif newDirection == Direction.LEFT:
                self.left = 1
                self.right = 0

    def turn_left(self):
        self.update_direction(Direction.LEFT)
        self.control.turnLeft()

    def turn_right(self):
        self.update_direction(Direction.RIGHT)
        self.control.turnRight()

    def stop(self):
        print("stop car")
        self.control.motorsStop()

    def turnOff(self):
        print("turning off car")
        self.control.cleanup()
    
    def accelerate(self):
        print("Accelerating car")
        self.update_direction(Direction.FORWARD)
        self.throttleHelper(1)

    def decelerate(self):
        print("Decelerating car")
        self.update_direction(Direction.BACK)
        self.throttleHelper(-1)

    def throttleHelper(self, newThrottle):
        print("in throttle helper")
        if self.telemetry.throttle + newThrottle < 4:
            print(f"changing speed to {self.telemetry.throttle + newThrottle}")
            self.telemetry.throttle += newThrottle
            self.control.motorsForward(self.telemetry.throttle)
        elif self.telemetry.throttle + newThrottle >= 4:
            print("RC_Car is at max speed")
            self.telemetry.throttle = 4
            self.control.motorsForward(self.telemetry.throttle)
        elif self.telemetry.throttle + newThrottle <= 0:
            print("RC_Car is at zero speed")
            self.telemetry.throttle = 0
            self.control.motorsStop()


class Direction(Enum):
    FORWARD = 0
    BACK = 1
    LEFT = 2
    RIGHT = 3
