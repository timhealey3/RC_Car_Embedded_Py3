import sched, time
import sys
sys.path.append('/home/timh/codingProjects/src/car')
from RC_Car import RC_Car

print("Setting up websocket")
rc_car = RC_Car()
model = NeuralNetwork()
model.load_state_dict(torch.load('../models/latest_model.pth', weights_only=True))
model.eval()

def get_data():
    print("Info - get data")
    return rc_car.telemetry.data()

def training_data():
    print("Info - Entering training data process")
    if rc_car.autonomous_mode:
        print("Info - Car is in autonomous mode, cannot enter training mode")
        # photo
        image = rc_car.camera.take_photo_auto()
        prediction = model.predict(image)
        print(prediction)
    else:
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

    if data.get('status') == "AUTO":
        print("car into autonomous mode")
        rc_car.autonomous_mode = True


    elif data.get('status') == 'NONE':
        if data.get('forward') == "FORWARD":
            print("Debug - Handle Message: Forward")
            rc_car.accelerate()
        if data.get('forward') == "BACKWARD":
            print("Debug - Handle Message: Stop")
            rc_car.decelerate()
        if data.get('turn') == "LEFT":
            print("Debug - Handle Message: Left")
            rc_car.turn_left()
        if data.get('turn') == "RIGHT":
            print("Debug - Handle Message: Right")
            rc_car.turn_right()
        if data.get('turn') == "STRAIGHTEN":
            print("Debug - Handle Message: Straighten")
            rc_car.straighten()
    else:
        print("Unknown command status")
