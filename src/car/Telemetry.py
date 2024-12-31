
class Telemetry:
    def __init__(self):
        self.on = True
        self.forward = False
        self.backward = False
        self.right = False
        self.left = False
        self.throttle = 0

    def get_training_data(self):
        return [self.forward, self.left, self.right]