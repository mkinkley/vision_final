import cv2
import tensorflow as tf
import numpy as np


def load_inference_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile("frozen_inference_graph.pb", 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return detection_graph, sess



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
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)


cam = cv2.VideoCapture(0)

cv2.namedWindow("win", cv2.WINDOW_NORMAL)
cv2.resizeWindow("win", 600, 600)
detection_graph, sess, = load_inference_graph()

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, scores = find_hand(frame, detection_graph, sess)
    draw_box_on_image(1, .2, scores, boxes, 640, 480, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("win", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        break

cam.release()
cv2.destroyAllWindows()
