import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import tensorflow as tf
import numpy as np
import detect
import time

cascPath = 'backend/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
detection_graph, sess, = detect.load_inference_graph()

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




def find_hand(image, graph, sess):
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores,
                detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    hands = []
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(top))
            p3 = (int(left), int(bottom))
            p4 = (int(right), int(bottom))
            hands.append((p1, p2, p3, p4))

            #cv2.rectangle(image_np, p1, p4, (77, 255, 9), 3, 1)
    return hands

def background_thread():
    capture = cv2.VideoCapture(0)

    global x1
    global y1
    global w1
    global frms
    global trackingFace
    while True:
       frms += 1
       ret, image = capture.read()
       height, width = image.shape[:2]
       image = cv2.flip(image, 1)
       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       boxes, scores = find_hand(image, detection_graph, sess)
       hands = draw_box_on_image(1, .2, scores,
                                 boxes, width, height, image)

       #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

       #facial detection
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
       # then you want to not send info over if what hands detected was actually a face
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
       if len(hands) > 0:
        one_hand = hands[0]
        top_left_x = one_hand[0][0]
        top_left_y = one_hand[0][1]
        top_right_x = one_hand[1][0]
        bottom_left_y = one_hand[2][1]
        length_y = bottom_left_y - top_left_y
        length_x = top_left_x - top_right_x
        area = length_x * length_y
        #if (abs(top_left_x - x1) > 20 and abs(top_left_y - y1) > 20):
        socketio.emit('message', {'x': top_left_x
                , 'y': top_left_y, 'area': area})
       cv2.imshow('Camera stream', image)

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


