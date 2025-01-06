import websocket
import json
import handle_message

def on_open(ws):
    print("Connected to server")
    message = {"status": "connected", "message": "Client has connected to the server"}
    ws.send(json.dumps(message))

#def telemetry_send(tele_data):
#    print(f"sending telemetry data {tele_data}")
#    socket.send(json.dumps(tele_data))

#def telemetry_get(ws):
#    print("sending telemetry data")
#    message = {"telemetry": tele}
#    socket.send(json.dumps(message))

def on_message(ws, message):
    data = json.loads(message)
    print(f"received from server: {data}")

def on_close(ws, close_status_code, close_msg):
    print("Disconnected from server")

socket = websocket.WebSocketApp(
    'ws://192.168.1.131:8090',
    on_open=on_open,
    on_message=on_message,
    on_close=on_close,
)

socket.run_forever()
