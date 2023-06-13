
import paho.mqtt.client as mqtt
import json

# a callback function
def on_message_emotion(client, userdata, msg):
    # Message is an object and the payload property contains the message data which is binary data.
    # The actual message payload is a binary buffer. 
    # In order to decode this payload you need to know what type of data was sent.
    # If it is JSON formatted data then you decode it as a string and then decode the JSON string as follows:
    decoded_message = str(msg.payload.decode("utf-8"))
    msg = json.loads(decoded_message)
    print(msg)
    # print('Received a new emotion data ', str(msg.payload))
    # print('message topic=', msg.topic)

# Give a name to this MQTT client
client = mqtt.Client('backend_server')
client.message_callback_add('emotion', on_message_emotion)

# IP address of your MQTT broker, using ipconfig to look up it  
client.connect('127.0.0.1', 1883)
# 'greenhouse/#' means subscribe all topic under greenhouse
client.subscribe('emotion')

client.loop_forever()