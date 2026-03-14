import cv2
import cvzone
import requests
from flask import Flask, request, render_template
import os

thres = 0.55
nmsThres = 0.2

classNames = []
classFile = 'object_det/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)

configPath = 'object_det/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'object_det/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Mock location for demonstration purposes
latitude = 37.7749
longitude = -122.4194

def send_location(lat, lon):
    # Replace with your actual endpoint and payload structure
    url = "http://example.com/api/send_location"
    payload = {
        "latitude": lat,
        "longitude": lon
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Location sent successfully")
    else:
        print("Failed to send location")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    images = os.listdir(UPLOAD_FOLDER)
    return render_template('index.html', images=images)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return 'File uploaded successfully'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classNames[classId - 1] == "person":
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, f'{classNames[classId - 1].upper()} {int(confidence * 100)}%',
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Count the number of detected victims
        victim_count = sum(1 for classId in classIds.flatten() if classNames[classId - 1] == "person")
        cv2.putText(img, f'Victims: {victim_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Send the live location of victims
        send_location(latitude, longitude)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()