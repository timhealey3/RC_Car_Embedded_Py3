from newCar import newCar

print("Setting up car")
my_car = newCar()

def incoming_data(data):
    print(f"Incoming data: {data}")
    if data.get('status') == "START":
        print("Car is starting")

    elif data.get('status') == 'NONE':
        if data.get('forward') == "FORWARD":
            my_car.accelerate(1)
        if data.get('forward') == "BACKWARD":
            my_car.accelerate(-1)
        if data.get('turn') == "LEFT":
            my_car.turn(1)
        if data.get('turn') == "RIGHT":
            my_car.turn(-1)
    else:
        print("Unknown command status")
