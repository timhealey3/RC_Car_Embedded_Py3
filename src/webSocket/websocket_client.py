import websocket
import json
import handle_message
from src.car.Telemetry import Telemetry


def on_open(ws):
    print("Connected to server")
    message = {"status": "connected", "message": "Client has connected to the server"}
    ws.send(json.dumps(message))

def telemetry_get(tele: Telemetry):
    print("sending telemetry data")
    message = {"telemetry": tele}
    socket.send(json.dumps(message))

def on_message(ws, message):
    data = json.loads(message)
    print(f"received from server: {data}")
    handle_message.incoming_data(data)

def on_close(ws, close_status_code, close_msg):
    print("Disconnected from server")

socket = websocket.WebSocketApp(
    'INPUT WEB SOCKET HERE',
    on_open=on_open,
    on_message=on_message,
    on_close=on_close,
)

socket.run_forever()
