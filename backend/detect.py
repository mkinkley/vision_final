import cv2
import tensorflow as tf
import numpy as np


def load_inference_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile("backend/frozen_inference_graph.pb", 'rb') as fid:
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
            
            cv2.rectangle(image_np, p1, p4, (77, 255, 9), 3, 1)




