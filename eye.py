import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model

IMG_SIZE = (24, 24)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 데이터를 더 모으자앗
model = load_model('models/2020_05_24_17_05_58.h5')

# model.summary()

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

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect


cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
  print("Unable to read camera")

while True:
    ret, img_ori = cap.read()
    if not ret:
      break
    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # TO DO 얼굴 인식 정확도 Up
    faces = detector(gray)

    for face in faces:
      shapes = predictor(gray, face)
      shapes = face_utils.shape_to_np(shapes)

      eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
      print(eye_img_l.shape)
      eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)


      eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

      pred_l = model.predict(eye_input_l)

      state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'

      state_l = state_l % pred_l


      cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
      cv2.rectangle(img, tuple(eye_rect_l[0:2]), tuple(eye_rect_l[2:4]), (255, 0, 0), 2)

      cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    cv2.imshow('Test', img)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
      break

cap.release()
cap.destroyAllWindows()