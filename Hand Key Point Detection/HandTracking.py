#dependencies
import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
prev_time,curr_time = 0,0
while True:
    _,img = cam.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,hand_landmarks,mpHands.HAND_CONNECTIONS)
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    cv2.putText(img,f'FPS:{str(int(fps))}',(400,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow('Frames',img)
    cv2.waitKey(1)