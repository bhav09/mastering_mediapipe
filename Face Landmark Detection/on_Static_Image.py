#dependencies
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

# Setup the face landmarks function for images.
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                         min_detection_confidence=0.5)


# Initialize the mediapipe drawing styles class.
mp_drawing_styles = mp.solutions.drawing_styles

sample_img = cv2.imread('Images/say-aa.jpg')

face_mesh_results = face_mesh_images.process(sample_img[:, :, ::-1])

# Get the list of indexes of the left and right eye.
LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
'''
# Check if facial landmarks are found.
if face_mesh_results.multi_face_landmarks:

    # Iterate over the found faces.
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):

        # Display the face number upon which we are iterating upon.
        print(f'FACE NUMBER: {face_no + 1}')
        print('-----------------------')

        # Display the face part name i.e., left eye whose landmarks we are gonna display.
        print(f'LEFT EYE LANDMARKS:\n')

        # Iterate over the first two landmarks indexes of the left eye.
        for LEFT_EYE_INDEX in LEFT_EYE_INDEXES[:2]:
            # Display the found normalized landmarks of the left eye.
            print(face_landmarks.landmark[LEFT_EYE_INDEX])

        # Display the face part name i.e., right eye whose landmarks we are gonna display.
        print(f'RIGHT EYE LANDMARKS:\n')

        # Iterate over the first two landmarks indexes of the right eye.
        for RIGHT_EYE_INDEX in RIGHT_EYE_INDEXES[:2]:
            # Display the found normalized landmarks of the right eye.
            print(face_landmarks.landmark[RIGHT_EYE_INDEX])
'''
img_copy = sample_img[:, :, ::-1].copy()

# Check if facial landmarks are found.
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
fig = plt.figure(figsize=[10, 10])

# Display the resultant image with the face mesh drawn.
plt.title("Resultant Image");
plt.axis('off');
plt.imshow(img_copy);
plt.show()
