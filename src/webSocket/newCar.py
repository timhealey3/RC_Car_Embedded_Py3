class newCar:
    def __init__(self, speed = 0, direction = 0):
        self.speed = speed
        self.direction = direction
        self.autonomous_mode = False
        self.training_mode = False

    def turn(self, newDirection):
        print("Turning car")
        self.direction += newDirection
        print(f"New direction: {self.direction}")
        turn_wheels()

    def accelerate(self, newSpeed):
        print("Accelerating car")
        self.speed += newSpeed
        print(f"New speed: {self.speed}")
        spin_wheels()

    def telemetry(self):
        print(f"Speed: {self.speed}")
        print(f"Direction: {self.direction}")
        return self.speed, self.direction

    def spin_wheels():
        print("spinning wheels")

    def turn_wheels():
        print("turning wheels")
