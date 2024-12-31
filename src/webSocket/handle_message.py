import logging
import sys
sys.path.append('/home/timh/codingProjects/src/car') 
from RC_Car import RC_Car

print("Setting up websocket")
rc_car = RC_Car()

def incoming_data(data):
    print(f"Debug - Handle Message: Incoming data: {data}")
    if data.get('status') == "START":
        print("Info - Handle Message: Car is starting")

    if data.get('status') == "OFF":
        print("Info - Handle Message: Car is starting Off procedure")
        rc_car.stop()

    if data.get('status') == "MANUAL":
        print("Info - Handle Message: Car is in manual mode")

    if data.get('status') == "TRAINING":
        print("Info - Handle Message: Car is in training mode")

    if data.get('status') == "AUTO":
            print("Info - Handle Message: Car is in auto mode")

    elif data.get('status') == 'NONE':
        if data.get('forward') == "FORWARD":
            print("Debug - Handle Message: Forward")
            rc_car.accelerate()
        if data.get('forward') == "BACKWARD":
            print("Debug - Handle Message: Stop")
            rc_car.stop()
        if data.get('turn') == "LEFT":
            print("Debug - Handle Message: Left")
            rc_car.turn_left()
        if data.get('turn') == "RIGHT":
            print("Debug - Handle Message: Right")
            rc_car.turn_right()
    else:
        print("Unknown command status")
