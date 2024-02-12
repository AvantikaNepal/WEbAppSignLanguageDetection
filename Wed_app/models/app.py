# Import necessary libraries
import Mycode
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from keras import load_model
import tensorflow


# Load the trained model and actions
model = Mycode.model
model.load_weights('April26.h5')
actions = np.array(['hello', 'iloveyou', 'thanks', 'bye', 'help','iamsleepy'])

# Create a Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Route to render the HTML file with the camera button
@app.route('/')
def index():
    return render_template('templates/index.html')

# Function to handle the camera start event
@socketio.on('start_camera')
def start_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()
        emit('video_frame', {'data': data})

# Function to handle the prediction event
@socketio.on('predict')
def predict(data):
    frame_data = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(frame_data, flags=1)
    image, results = Mycode.mediapipe_detection(frame, Mycode.holistic)
    keypoints = Mycode.extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        prediction = actions[np.argmax(res)]
        emit('prediction_result', {'prediction': prediction})

if __name__ == '__main__':
    # Start the Flask app with SocketIO
    socketio.run(app)