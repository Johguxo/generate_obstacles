import cv2 
from imageai.Detection import ObjectDetection
from imageai.Detection.Custom import CustomObjectDetection
import math

import numpy as np
import os as os
import random

def to_txt(lines):
    path = '/home/johann/ws_moveit/src/thor_arm_pick_place/src/'
    file = 'bottles_position.txt'
    with open(path+file,'w') as f:
        for line in lines:
            f.write(str(line['x']) + ' ' + 
                    str(line['y']) + ' ' + 
                    str(line['z'])+'\n')

def open_images():
    file = 'botella_brazo.jpg'
    file_output = 'botella-predict.jpg'
    img1 = cv2.imread(file)
    img2 = cv2.imread(file_output)
    img1 = cv2.resize(img1,(360,480))
    img2 = cv2.resize(img2,(360,480))
    Hori = np.concatenate((img1, img2), axis=1)
    while True:
        cv2.imshow('Imagenes Botellas', Hori)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cv2.destroyAllWindows()

def showImage(img):
    window_name = 'image'
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("botellas_model.h5")
detector.setJsonPath("detection_config_bottle2.json")
detector.loadModel()


cap = cv2.VideoCapture(0)
address = "http://192.168.1.15:8080/video"
cap.open(address)


if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    
    filename = './botella_brazo.jpg'
  
    cv2.imwrite(filename, frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()



detections = detector.detectObjectsFromImage(input_image="botella_brazo.jpg", output_image_path="botella-predict.jpg",minimum_percentage_probability=10)
#detections = detector.detectObjectsFromImage(input_image="094.jpg", output_image_path="094-predict.jpg")
lines_position = []
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    #print( type(detection["box_points"]) )
    i =   (detection["box_points"][0]+detection["box_points"][2])/2 
    j =   (detection["box_points"][1]+detection["box_points"][3])/2

    ##para coordenadas
    d = 89
    fh = 52
    fv = 42
    height = 720
    width = 960
    x = d * (i/(width-1)-0.5)*math.tan((fh*math.pi/180))*1.1
    y = d * (0.5-j/(height-1))*math.tan((fv*math.pi/180))*1.1
    z_ref = random.randint(-200,200)
    z = 15 + 0.01*z_ref
    print('Pixeles:', j,i,'Posicion:',x,y,z)
    lines_position.append({'x':x,'y':y,'z':z})

to_txt(lines_position)
open_images()

#botellasOnly = detector.CustomObjects(bottle=True)
#detectedImage, detections = detector.detectCustomObjectsFromImage(custom_objects=botellasOnly, output_type="array", input_image="ImageCamera/{0}".format(randomFile), minimum_percentage_probability=20)
#onvertedImage = cv2.cvtColor(detectedImage, cv2.COLOR_RGB2BGR)
#howImage(convertedImage)

