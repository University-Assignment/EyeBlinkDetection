import socket
import cv2, dlib
import numpy as np
from keras.models import load_model
from imutils import face_utils
from _thread import *

def threaded(client_socket):
    count = 0
    closed = False
    while True:
        message = '1'
        client_socket.send(message.encode())

        length = recvall(client_socket, 16)
        stringData = recvall(client_socket, int(length))
        data = np.frombuffer(stringData, dtype='uint8')

        img_ori = cv2.imdecode(data, 1)

        img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

        img = img_ori.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # TO DO 얼굴 인식 정확도 Up
        faces = detector(gray)

        for face in faces:
            shapes = predictor(gray, face)
            shapes = face_utils.shape_to_np(shapes)

            eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
            eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

            eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
            eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)

            # cv2.imshow('left eye', eye_img_l)
            # cv2.imshow('right eye', eye_img_r)

            eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
            eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

            pred_l = model.predict(eye_input_l)
            pred_r = model.predict(eye_input_r)

            state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
            state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

            state_l = state_l % pred_l
            state_r = state_r % pred_r

            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
            cv2.rectangle(img, tuple(eye_rect_l[0:2]), tuple(eye_rect_l[2:4]), (255, 0, 0), 2)
            cv2.rectangle(img, tuple(eye_rect_r[0:2]), tuple(eye_rect_r[2:4]), (255, 0, 0), 2)

            cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

            if pred_l > 0.9 and not closed:
                closed = True
                count += 1
                print("눈 깜빡임", count, "번")
            elif pred_l <= 0.1:
                closed = False
            cv2.imshow('Image', img)

        key = cv2.waitKey(1)
        if key == 27:
            break
    client_socket.close()

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def crop_eye(img, eye_points): # 눈 추출
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# TO DO 오른쪽 눈에 대한 모델 작성
model = load_model('model.h5')

HOST = '127.0.0.1'
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

print('server start')

while True:
    print('wait')

    client_socket, addr = server_socket.accept()
    start_new_thread(threaded, (client_socket, ))

server_socket.close()