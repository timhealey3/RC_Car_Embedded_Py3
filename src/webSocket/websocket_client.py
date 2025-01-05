import websocket
import json
import handle_message


def on_open(ws):
    print("Connected to server")
    message = {"status": "connected", "message": "Client has connected to the server"}
    ws.send(json.dumps(message))

def telemetry_send(tele_data):
    print(f"sending telemetry data {tele_data}")
    socket.send(json.dumps(tele_data))

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
