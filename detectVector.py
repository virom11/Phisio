#!/usr/bin/python

import math
import numpy as np
import scipy
import os
import sys
import scipy.misc
import scipy.cluster
from PIL import Image, ImageDraw
import dlib
import cv2
import openface

import scriptsVector

def asymmetry(predictor_model,file_name):
    pose_landmarks=scriptsVector.face_aligner_func(predictor_model,file_name)
    i=0
    asymmetry=0
    while i<68:
        if((i>16) and (i!=27) and (i!=33)):
            #print("Точка "+str(i)+":" + str((pose_landmarks.part(i).y-pose_landmarks.part(27).y)*(pose_landmarks.part(33).x-pose_landmarks.part(27).x)-(pose_landmarks.part(33).y-pose_landmarks.part(27).y)*(pose_landmarks.part(i).x-pose_landmarks.part(27).x)))
            asymmetry+=(pose_landmarks.part(i).y-pose_landmarks.part(27).y)*(pose_landmarks.part(33).x-pose_landmarks.part(27).x)-(pose_landmarks.part(33).y-pose_landmarks.part(27).y)*(pose_landmarks.part(i).x-pose_landmarks.part(27).x)
        i+=1
    if(asymmetry>0):
        left = 100
        right = 0
    else:
        left = 0
        right = 100
    #print(asymmetry)
    return right, left

def nose(predictor_model,file_name,pose_landmarks):
    im = Image.open(file_name) # Can be many different formats.
    pix = im.load()
    #print('Image Size: '+str(im.size))  # Get the width and hight of the image for iterating over
    #pose_landmarks=scriptsVector.face_aligner_func(predictor_model,file_name)
    limit_y1=round(pose_landmarks.part(27).y)
    #print('limit_y1: '+str(limit_y1))
    limit_y2=round(2*pose_landmarks.part(27).y-pose_landmarks.part(28).y)
    #print('limit_y2: '+str(limit_y2))
    x1=pose_landmarks.part(27).x
    x2=pose_landmarks.part(33).x
    y1=pose_landmarks.part(27).y
    y2=pose_landmarks.part(33).y
    y=round(pose_landmarks.part(27).y)
    first_color=pix[round(((y-y1)*(x2-x1)+x1*(y2-y1))/(y2-y1)),y]
    print('first_color: '+str(first_color))
    average_f=(first_color[0]+first_color[1]+first_color[2])/3
    print('average_f: '+str(average_f))
    #print('x1: '+str(x1))
    #print('x2: '+str(x2))
    #print('y1: '+str(y1))
    #print('y2: '+str(y2))
    counter=0
    sum_avs=0
    while((y>limit_y2) and (y>0)):
        counter+=1
        pix_x=round(((y-y1)*(x2-x1)+x1*(y2-y1))/(y2-y1))
        second_color=pix[pix_x,y]
        average_s=(second_color[0]+second_color[1]+second_color[2])/3
        sum_avs+=average_s
        #print('pix_x: '+str(pix_x))
        #print('second_color: '+str(second_color))
        #print(str(y)+" "+str(average_s))
        # Get the RGBA Value of the a pixel of an image
        #pix[pix_x,y] = (255,255,255)  # Set the RGBA Value of the image (tuple)
        y-=1
    #im.save(file_name)
    sum_avs=sum_avs/counter
    print('sum_avs: '+str(sum_avs))
    
    if(sum_avs>(average_f-10)):
        straight=100
    elif(sum_avs<(average_f-16)):
        straight=0
    else:
        straight=(average_f-sum_avs-10)/0.06

    unstraight=100-straight
    return straight,unstraight