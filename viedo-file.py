import pandas as pd
import numpy as np
import mediapipe as mp
import joblib
import os,cv2,math,glob,random
# import scipy.io as sio
from math import cos, sin
from pathlib import Path

modle = joblib.load('/home/mahmoud/Desktop/iti/pose-estimation-project/my_model.pkl')



def draw_axis(img, pitch,yaw,roll,tdx=None, tdy=None, size = 100):

    yaw = -yaw
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # shape = img.shape 
    # tdx = int(x * shape[1])
    # tdy = int(y * shape[0])
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3) # red
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3) # green
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2) # blue

    return img # cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if i read the file w


def gen_landmarks(image):
    faceModule = mp.solutions.face_mesh
    data_list = []
    count = 0
    with faceModule.FaceMesh(static_image_mode=True) as faces:
            # image = plt.imread(im)
            results = faces.process(image)
            if results.multi_face_landmarks != None:
                face = results.multi_face_landmarks[0]
                for landmark in face.landmark:
                    data_list.append(landmark.x)
                    data_list.append(landmark.y)
                    if count == 97:
                        rx = landmark.x * image.shape[1]
                        ry =  landmark.y * image.shape[0]
                    count += 1
    return np.array(data_list).reshape((1,-1)), rx,ry

# img = cv2.imread('AFLW2000/image03227.jpg')
# cv2.imshow('window', img)
# cv2.waitKey(0)


cam = cv2.VideoCapture(0)
vid_cod = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter("/home/mahmoud/Desktop/iti/pose-estimation-project/cam2_video.mp4", vid_cod, 10.0, (640,480))

while cam.isOpened():
    success, image = cam.read()
    if not success:
        print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
        continue
    else:
        # lmrks = gen_landmarks(image)
        try:
            lmks ,rx,ry= gen_landmarks(image)
            p,y, r = modle.predict(lmks)[0]
            pimage = draw_axis(image,p,y, r,rx,ry)
            cv2.imshow('cam',pimage)


            output.write(pimage)
        except:
            break
            # output.write(image)
            # continue
            # im = cv2.imread('/home/mahmoud/Pictures/my-pic.jpg')
            # cv2.imshow('cam',im)
            # output.write(image)
        # print('hellow')
        # lmks ,rx,ry= gen_landmarks(image)
        # y,p, r = gs.predict(lmks)[0]
        # plt.imshow(draw_axis(my_pic,y,p, r,rx,ry))

    if cv2.waitKey(1) &0XFF == ord('q'):
        break

cam.release()
output.release()
cv2.destroyAllWindows()