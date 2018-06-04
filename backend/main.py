import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time

cascPath = '../data//haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
thread = None
x1 = 0
w1 = 0
y1 = 0
trackingFace = 0
frms = 0
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]
if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()


def background_thread():
    capture = cv2.VideoCapture(0)
    global x1
    global y1
    global w1
    global frms
    global trackingFace
    while True:
       frms += 1
       #f not trackingFace or frms % 50 == 0:
       ret, image = capture.read()
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       faces = faceCascade.detectMultiScale(
            gray, 1.3, 5
       )
       maxArea = 0
       x = 0
       y = 0
       w = 0
       h = 0

       # Loop over all faces and check if the area for this
       # face is the largest so far
       for (_x, _y, _w, _h) in faces:
          if _w * _h > maxArea and _w > 0 and _h > 0:
            x = int(_x)
            y = int(_y)
            w = int(_w)
            h = int(_h)
            maxArea = w * h
       if maxArea > 0:
         bbox = (x + 500, y, w - 10, h - 10)
         x1 = x + 500
         y1 = y
         w1 = w
         #cv2.rectangle(frame, p1, p2, (249, 151, 208), 2, 1)
       cv2.imshow('Camera stream', image)
       socketio.emit('message', {'x': x1, 'y': y1, 'area': w1})
       time.sleep((1000.0 / 30) / 1000)

@socketio.on('connect')
def connect():
    global thread
    if thread is None:
        thread = socketio.start_background_task(target=background_thread)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)


