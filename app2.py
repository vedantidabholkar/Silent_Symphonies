from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import DepthwiseConv2D  # Import DepthwiseConv2D

# Initialize Flask app
app = Flask(__name__)

# Initialize camera and sign language detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Define a custom DepthwiseConv2D if necessary
def custom_depthwise_conv2d(*args, **kwargs):
    if 'groups' in kwargs:
        del kwargs['groups']  # Ignore the 'groups' argument
    return DepthwiseConv2D(*args, **kwargs)

get_custom_objects().update({'DepthwiseConv2D': custom_depthwise_conv2d})

# Load the classifier model and labels
classifier = Classifier(r"C:\Users\himan\OneDrive\Desktop\Silent Symphonies Dynamic\keras_model_3.h5", 
                        r"C:\Users\himan\OneDrive\Desktop\Silent Symphonies Dynamic\labels_3.txt")

# Predefined labels
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

offset = 20
imgSize = 300

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Drawing the bounding box and label on the output image
                cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            # Encode frame to byte format
            ret, buffer = cv2.imencode('.jpg', imgOutput)
            imgOutput = buffer.tobytes()

            # Yield frame in a byte format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + imgOutput + b'\r\n')

@app.route('/')
def index():
    # Render the HTML template
    return render_template('index_f.html')

@app.route('/video_feed')
def video_feed():
    # Route to provide the video stream
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(port=5002,debug=True)
