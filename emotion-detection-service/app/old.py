import numpy as np
import cv2
import time
import onnx
import onnxruntime as rt
import os
import mediapipe as mp
import paho.mqtt.client as mqtt
import json

client = mqtt.Client(client_id='emotion-detection-service')

mqtt_hostname = os.environ.get('MQTT_HOSTNAME')

client.connect(host = mqtt_hostname, port = 1883)

client.loop_start()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cv2.ocl.setUseOpenCL(False)

def detect_face(frame):
    bounding_boxes = []
    results = detector.process(frame)
    if results.detections:
        height,width,_ = frame.shape
        for detection in results.detections:
            bb = detection.location_data.relative_bounding_box
            x1 = max(0,bb.xmin*width)
            y1 = max(0,bb.ymin*height)
            x2 = min(width,(bb.xmin+bb.width)*width)
            y2 = min(height,(bb.ymin+bb.height)*height)
            bbox = np.array([x1,y1,x2,y2])
            bounding_boxes.append(bbox)

    return bounding_boxes

def preprocess(img):
    x = cv2.resize(img,(260,260))/255
    x[..., 0] = (x[..., 0]-0.485)/0.229
    x[..., 1] = (x[..., 1]-0.456)/0.224
    x[..., 2] = (x[..., 2]-0.406)/0.225
    return x.transpose(2, 0, 1).astype("float32")[np.newaxis,...]

def predict_emotions(face_img):
    scores = m.run(None,{"input": preprocess(face_img)})[0][0]
    x = scores
    pred = np.argmax(x)
    return label[pred], pred

model = "./model/enet_b2_8.onnx"
onnxModel = onnx.load(model)

output_names = [n.name for n in onnxModel.graph.output]

# dictionary which assigns each label an emotion (alphabetical order)
label = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}

m = rt.InferenceSession(model, providers=['CPUExecutionProvider'])

previous_time = int(time.time())
threshold = 1

cap = cv2.VideoCapture(0)
while True:

    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    
    if not ret:
        break

    current_time = int(time.time())
    if current_time - previous_time >= threshold:

        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        faces = detect_face(frame)

        for face in faces:
            face = face.astype(int)
            x1, y1, x2, y2 = face[0:4]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            img = frame[y1:y2,x1:x2,:]
            emotion, scores = predict_emotions(img)

            epoch_time = int(time.time())
            
            res = {
                'emotion': emotion,
                'time': current_time
            }
            
            info = client.publish(topic = "emotion", payload = json.dumps(res))
            info.wait_for_publish()
            #print(emotion, info.is_published())
            cv2.putText(frame, emotion, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        previous_time = current_time

    cv2.imshow('Video', cv2.resize(frame,(1280,720),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()