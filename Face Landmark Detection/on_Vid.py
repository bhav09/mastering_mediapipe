import cv2
from PIL import Image
import numpy as np
import mediapipe as mp
import time
import itertools
import matplotlib.pyplot as plt

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh

# Setup the face landmarks function for videos.
face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2,
                                         min_detection_confidence=0.5, min_tracking_confidence=0.3)

# Initialize the mediapipe drawing styles class.
mp_drawing_styles = mp.solutions.drawing_styles

#reading via webcam
cap = cv2.VideoCapture(0)
while True:
    _,img = cap.read()
    img_copy = cv2.flip(img,1)
    img = cv2.flip(img, 1)
    face_mesh_results = face_mesh_videos.process(img_copy[:, :, ::-1])
    if face_mesh_results.multi_face_landmarks:

        # Iterate over the found faces.
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            # Draw the facial landmarks on the copy of the sample image with the
            # face mesh tesselation connections using default face mesh tesselation style.
            mp_drawing.draw_landmarks(image=img_copy,
                                      landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # Draw the facial landmarks on the copy of the sample image with the
            # face mesh contours connections using default face mesh contours style.
            mp_drawing.draw_landmarks(image=img_copy, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    # Specify a size of the figure.
    stack = np.hstack([img,img_copy])
    cv2.imshow('Output',stack)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()