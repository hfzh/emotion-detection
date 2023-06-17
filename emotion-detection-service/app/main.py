import numpy as np
import cv2
import time
import os
import mediapipe as mp
import paho.mqtt.client as mqtt
import json
import torch
from torchvision import transforms
import timm
from PIL import Image

# client = mqtt.Client(client_id='emotion-detection-service')
# mqtt_hostname = os.environ.get('MQTT_HOSTNAME')
# client.connect(host = mqtt_hostname, port = 1883)
# client.loop_start()

device = "cpu"
img_size = 260
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

def get_probab(features):
    x = np.dot(features,np.transpose(classifier_weights)) + classifier_bias
    return x

def extract_features(face_img):
    img_tensor = test_transforms(Image.fromarray(face_img))
    img_tensor.unsqueeze_(0)
    features = model(img_tensor.to(device))
    features = features.data.cpu().numpy()
    return features
    
def predict_emotions(face_img):
    features = extract_features(face_img)
    scores = get_probab(features)[0]
    x = scores
    pred = np.argmax(x)
    return idx_to_class[pred],scores

idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}

test_transforms = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

path = "./assets/enet_b2_8_best.pt"
model = torch.load(path, map_location = torch.device(device))
classifier_weights = model.classifier.weight.cpu().data.numpy()
classifier_bias = model.classifier.bias.cpu().data.numpy()
model.classifier = torch.nn.Identity()
model = model.to(device)
model = model.eval()
print(path, test_transforms)


previous_time = int(time.time())
threshold = 1

cap = cv2.VideoCapture(0)
while True:
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
            print(res) 
            #info = client.publish(topic = "emotion", payload = json.dumps(res))
            #info.wait_for_publish()
            #print(emotion, info.is_published())
            # cv2.putText(frame, emotion, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        previous_time = current_time

    # cv2.imshow('Video', cv2.resize(frame,(1280,720),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
