import sched, time
import sys
sys.path.append('/home/timh/codingProjects/src/car')
from RC_Car import RC_Car

print("Setting up websocket")
rc_car = RC_Car()

def get_data():
    print("Info - get data")
    return rc_car.telemetry.data()

def training_data():
    print("Info - Entering training data process")
    rc_car.training()

def incoming_data(data):
    print(f"Debug - Handle Message: Incoming data: {data}")
    
    if data.get('status') == "START":
        print("Info - Handle Message: Car is starting")
        
    if data.get('status') == "OFF":
        print("Info - Handle Message: Car is starting Off procedure")
        rc_car.stop()
        rc_car.turnOff()

    if data.get('status') == "MANUAL":
        print("Info - Handle Message: Car is in manual mode")

    if data.get('status') == "TRAINING":
        print("Info - Handle Message: Car is in training mode")
        rc_car.training_mode = True

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
