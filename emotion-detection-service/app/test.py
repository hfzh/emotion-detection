import paho.mqtt.client as mqtt
import json

def on_message_emotion(client, userdata, msg):
    decoded_message = str(msg.payload.decode("utf-8"))
    msg = json.loads(decoded_message)

    print(msg, flush=True)

client = mqtt.Client('backend_server')
client.message_callback_add('emotion', on_message_emotion)
client.connect('127.0.0.1', 1883)
client.subscribe('emotion')

client.loop_forever()
