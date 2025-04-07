import websocket
import json
import handle_message
import threading
import time

ws = None

def training_send():
    print("training data send")
    handle_message.training_data()
    threading.Timer(.25, training_send).start()

def on_open(ws_param):
    global ws
    ws = ws_param
    print("Connected to server")
    message = {"status": "connected", "message": "Client has connected to the server"}
    ws.send(json.dumps(message))

def telemetry_send(tele_data):
    print(f"sending telemetry data {tele_data}")
    global ws
    if ws and ws.sock and ws.sock.connected:
        ws.send(json.dumps(tele_data))

def on_message(ws, message):
    data = json.loads(message)
    print(f"received from server: {data}")
    handle_message.incoming_data(data)
    telemetry_send(handle_message.get_data())

def on_close(ws, close_status_code, close_msg):
    print("Disconnected from server")

# Check if photo needs to be taken every 1 second in a separate thread
training_send()

socket = websocket.WebSocketApp(
    'data here',
    on_open=on_open,
    on_message=on_message,
    on_close=on_close,
)

# Start the WebSocket in a separate thread
socket_thread = threading.Thread(target=socket.run_forever)
socket_thread.start()
