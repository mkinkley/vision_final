import cv2
import numpy as np
import detect


face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface._default.xml')


def draw_flow(img, flow, x_off, y_off, step=16):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x + x_off, y + y_off, x+fx+x_off, y+fy+y_off]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    return img

cascPath = 'opencv/data/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
detection_graph, sess, = detect.load_inference_graph()


cam = cv2.VideoCapture(0)
cv2.namedWindow("win")
# cv2.namedWindow("test")
cv2.resizeWindow("win", 600, 600)


prev_frame = None
prev_feat = None
prev_points = None 

ball_x = 320
ball_y = 240
ball_vx = 0
ball_vy = 0
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes, scores = detect.find_hand(frame, detection_graph, sess)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    hands = detect.draw_box_on_image(1, .1, scores, boxes, frame.shape[1], frame.shape[0], frame, faces)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_images = []
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    for hand in hands:
        hand_images.append(gray[hand[0][1]:hand[3][1], hand[0][0]:hand[3][0]])
        x1 = hand[0][0]
        y1 = hand[0][1]
        x2 = hand[3][0]
        y2 = hand[3][1]

    if prev_points is not None:
        dx = x1 - prev_points[0][0]
        dy = y1 - prev_points[0][1]
        if ball_x >= x1 and ball_x <= x2 and ball_y >= y1 and ball_y <= y2:
            if abs(dx) < 50:
                ball_vx = dx
            if abs(dy) < 50:
                ball_vy = dy
        # print("dx: ", x1 - prev_points[0][0])
        # print("dy: ", y1 - prev_points[0][1])
    if len(hand_images) > 0:
        hand_img = hand_images[0]

        # feat = cv2.goodFeaturesToTrack(hand_img, 100, .1, 5)

        if prev_frame is not None:

            
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[y1:y2, x1:x2, 0], flow[y1:y2, x1:x2, 1])
            draw_flow(frame, flow[y1:y2, x1:x2, :], x1, y1)
            # cv2.imshow("test", flow)



    ball_x += ball_vx
    ball_y += ball_vy
    # if ball_vx > 0:
    #     ball_vx -= 1 
    # if ball_vy > 0:
    #     ball_vy -= 1 
    # if ball_vx < 0:
    #     ball_vx += 1 
    # if ball_vy < 0:
    #     ball_vy += 1 
    if ball_x <= 0 or ball_x >= 640:
        ball_vx *= -1
    if ball_y <= 0 or ball_y >= 480:
        ball_vy *= -1
    cv2.circle(frame, (ball_x, ball_y), 10, (255,0,0), -1)
    cv2.imshow("win", frame)
    prev_frame = gray 
    prev_points = ((x1, y1), (x2, y2))
    k = cv2.waitKey(1)
    if k%256 == 27:
        break

cam.release()
cv2.destroyAllWindows()
