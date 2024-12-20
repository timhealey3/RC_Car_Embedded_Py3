from src.car import RC_Car

print("Setting up car")
rc_car = RC_Car()

def incoming_data(data):
    print(f"Incoming data: {data}")
    if data.get('status') == "START":
        print("Car is starting")

    elif data.get('status') == 'NONE':
        if data.get('forward') == "FORWARD":
            rc_car.accelerate(1)
        if data.get('forward') == "BACKWARD":
            rc_car.accelerate(-1)
        if data.get('turn') == "LEFT":
            rc_car.turn(1)
        if data.get('turn') == "RIGHT":
            rc_car.turn(-1)
    else:
        print("Unknown command status")
