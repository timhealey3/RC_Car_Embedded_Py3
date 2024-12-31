# RC Car

Client Control Software Hosted: https://github.com/timhealey3/rc_car_websocket

This is a current ongoing project. I am building a rc car (not actually remote control but you get the idea) from scratch.
Currently the base car functionality works. I am now working on adding the CNN. Which includes building out a training mode
which will automatically take pictures every second and store it with the telemetry data.

## Running Program
Make sure Control.py has the correct pins for your pi. Change the websocket_client.py
websocket address to the ip address you are running the Kotlin GUI on. The Client Control
software needs to be started on that machine. Once that is done, running the program is as easy as 
running `python3 websocket_client.py`

### Hardware Needed to Build:
- Raspberry Pi 4
- Raspberry Pi Camera (this will be used to add machine vision at a later point)
- 4 DC motors
- 2 motor controllers
- 4 wheels
- External power supply

### Software: 
- Client Control Software with GUI - Kotlin
- Embedded Systems Car Control - Python3
