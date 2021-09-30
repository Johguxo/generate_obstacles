import numpy as np
import os as os
import random
import cv2

path = '/home/johann/ws_moveit/src/thor_arm_pick_place/src/'
file = 'text_test.txt'

def to_txt(lines):
    with open(path+file,'w') as f:
        for line in lines:
            f.write(str(line['x']) + ' ' + 
                    str(line['y']) + ' ' + 
                    str(line['z'])+'\n')
def operation_test_txt():
    lines = []
    for i in range(5):

        lines.append({'x':29.33+i,'y':15.6+i,'z':16+i})
    to_txt(lines)

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
open_images()