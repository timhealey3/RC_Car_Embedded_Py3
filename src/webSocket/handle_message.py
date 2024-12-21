import logging
import sys
sys.path.append('/home/timh/codingProjects/src/car') 
from RC_Car import RC_Car

print("Setting up websocket")
rc_car = RC_Car()

def incoming_data(data):
    print(f"Incoming data: {data}")
    if data.get('status') == "START":
        print("Car is starting")

    elif data.get('status') == 'NONE':
        if data.get('forward') == "FORWARD":
            rc_car.accelerate()
        if data.get('forward') == "BACKWARD":
            rc_car.stop()
        if data.get('turn') == "LEFT":
            rc_car.turnLeft()
        if data.get('turn') == "RIGHT":
            rc_car.turnRight()
    else:
        print("Unknown command status")
