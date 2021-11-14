
# Dependencies
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import numpy as np
import time

#Object for capturing via Webcam
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
segment_obj = SelfiSegmentation()
count = 0
list_imgs = os.listdir('Images')
ptime = 0

#Capturing image frames continously
while True:
    #Reading frames
    _,img = cam.read()
    ctime = time.time()
    imgbg = cv2.imread(f'Images/{list_imgs[count % len(list_imgs)]}')
    
    #flipping the image horizontally
    img = cv2.flip(img,1)
    
    #removing background
    output = segment_obj.removeBG(img,imgbg,threshold=0.3)
    
    #stacking image wtih the original one
    final = np.hstack([img,output])
    
    #calculating FPS
    fps = 1/(ctime-ptime)
    ptime = ctime
    
    #adding text to the main window
    cv2.putText(final,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)
    cv2.imshow('Window',final)
    
    # a for the next background and d for the previous
    key = cv2.waitKey(1)
    if key == ord('a'):
        count += 1
    elif key == ord('d'):
        count -= 1
    elif key == ord('q'):
        break
