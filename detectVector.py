#!/usr/bin/python
'''
--------------------------------------------------------------------
def asymmetry(predictor_model,file_name):
def nose(predictor_model,file_name,pose_landmarks):
def nose_size(predictor_model,file_name,pose_landmarks):
def nose_wings(predictor_model,file_name,pose_landmarks):
def hump_nose(predictor_model,file_name,pose_landmarks):
def forehead(predictor_model,file_name,pose_landmarks):
def eyelids(predictor_model,file_name,pose_landmarks):
def hair_color(predictor_model,file_name,pose_landmarks):
--------------------------------------------------------------------
'''

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
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scriptsVector
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

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
    limit_y1=round(pose_landmarks.part(27).y)
    limit_y2=round(2*pose_landmarks.part(27).y-pose_landmarks.part(28).y)
    x1=pose_landmarks.part(27).x
    x2=pose_landmarks.part(33).x
    y1=pose_landmarks.part(27).y
    y2=pose_landmarks.part(33).y
    y=round(pose_landmarks.part(27).y)
    first_color=pix[round(((y-y1)*(x2-x1)+x1*(y2-y1))/(y2-y1)),y]
    #print('first_color: '+str(first_color))
    average_f=(first_color[0]+first_color[1]+first_color[2])/3
    #print('average_f: '+str(average_f)) 

    counter=0
    sum_avs=0
    while((y>limit_y2) and (y>0)):
        counter+=1
        pix_x=round(((y-y1)*(x2-x1)+x1*(y2-y1))/(y2-y1))
        second_color=pix[pix_x,y]
        average_s=(second_color[0]+second_color[1]+second_color[2])/3
        sum_avs+=average_s
        # Get the RGBA Value of the a pixel of an image
        #pix[pix_x,y] = (255,255,255)  # Set the RGBA Value of the image (tuple)
        y-=1
    #im.save(file_name)
    sum_avs=sum_avs/counter
    #print('sum_avs: '+str(sum_avs))
    
    if(sum_avs>(average_f-10)):
        straight=100
    elif(sum_avs<(average_f-16)):
        straight=0
    else:
        straight=(average_f-sum_avs-10)/0.06

    unstraight=100-straight
    return straight,unstraight

def nose_size(predictor_model,file_name,pose_landmarks):
    big = 0
    small = 0
    tip=0
    dist=scriptsVector.distance(pose_landmarks.part(29).x,pose_landmarks.part(29).y,pose_landmarks.part(30).x,pose_landmarks.part(30).y)
    dist_control=scriptsVector.distance(pose_landmarks.part(30).x,pose_landmarks.part(30).y,pose_landmarks.part(33).x,pose_landmarks.part(33).y)
    #print('Distance: '+ str(dist))
    #print('Control distance: '+ str(dist_control))

    if(dist_control==dist):
        big=50
        small=50
    elif(dist_control<dist):
        big=50+(dist-dist_control)/0.2
        small=100-big
    elif(dist_control>dist):
        small=50+(dist_control-dist)/0.2
        big=100-small
    #print('big: '+ str(big))
    #print('small: '+ str(small))
    big,small=scriptsVector.range(big,small)

    if(big==100):
        tip=100
    
    return big,small,tip

def nose_wings(predictor_model,file_name,pose_landmarks):
    nose_wings=0
    im = Image.open(file_name) # Can be many different formats.
    pix = im.load()
    a=27
    b=30
    distance_1=scriptsVector.distance(pose_landmarks.part(27).x,pose_landmarks.part(27).y,pose_landmarks.part(0).x,pose_landmarks.part(0).y)
    distance_2=scriptsVector.distance(pose_landmarks.part(27).x,pose_landmarks.part(27).y,pose_landmarks.part(16).x,pose_landmarks.part(16).y)
    if(distance_1>distance_2):
        a=31
        b=39
    else:
        a=35
        b=42

    limit_y1=round(pose_landmarks.part(30).y+((pose_landmarks.part(29).y-pose_landmarks.part(30).y)*0.35))
    limit_y2=round(pose_landmarks.part(28).y+((pose_landmarks.part(28).y-pose_landmarks.part(29).y)*0.25))
    x1=pose_landmarks.part(a).x
    x2=pose_landmarks.part(b).x
    y1=pose_landmarks.part(a).y
    y2=pose_landmarks.part(b).y
    y=limit_y1
    first_color=pix[round(((y-y1)*(x2-x1)+x1*(y2-y1))/(y2-y1)),y]
    average_f=(first_color[0]+first_color[1]+first_color[2])/3
    flag=0
    minimal=average_f
    maximum=0
    while((y>limit_y2) and (y>0)):
        pix_x=round(((y-y1)*(x2-x1)+x1*(y2-y1))/(y2-y1))
        second_color=pix[pix_x,y]
        average_s=(second_color[0]+second_color[1]+second_color[2])/3
        #print(str(average_s))
        if(average_s<minimal):
            minimal=average_s
            flag=y
            maximum=average_s
        if(average_s>maximum):
            maximum=average_s

        # Get the RGBA Value of the a pixel of an image
        #pix[pix_x,y] = (255,255,255)  # Set the RGBA Value of the image (tuple)
        y-=1
    #print('minimal: '+str(minimal))
    #print('maximum: '+str(maximum))
    if(minimal<(maximum-40)):
        nose_wings=100
    else:
        nose_wings=(maximum-minimal)/0.4
    #im.save(file_name)

    return nose_wings

def hump_nose(predictor_model,file_name,pose_landmarks):

    hump_nose = 0
    im = Image.open(file_name) # Can be many different formats.
    pix = im.load()
    distance_1=scriptsVector.distance(pose_landmarks.part(27).x,pose_landmarks.part(27).y,pose_landmarks.part(0).x,pose_landmarks.part(0).y)
    distance_2=scriptsVector.distance(pose_landmarks.part(27).x,pose_landmarks.part(27).y,pose_landmarks.part(16).x,pose_landmarks.part(16).y)
    #print('distance_1: '+str(distance_1))
    #print('distance_2: '+str(distance_2))
    
    #round(((y-y1)*(x2-x1)+x1*(y2-y1))/(y2-y1))
    if(distance_1>distance_2):
        a=42
        sign=1
    else:
        a=39
        sign=-1
    b=28
    s=[]
    #print('a: '+str(a))
    while(b<30):
        #print('b: '+str(b))
        x1=pose_landmarks.part(27).x
        x2=pose_landmarks.part(30).x
        y1=pose_landmarks.part(27).y
        y2=pose_landmarks.part(30).y
        y=round(pose_landmarks.part(b).y)
        limit_x1=round(pose_landmarks.part(b).x)
        limit_x2=round(((y-y1)*(x2-x1)+pose_landmarks.part(a).x*(y2-y1))/(y2-y1))
        x=limit_x1
        first_color=pix[x,y]
        #print('first_color: '+str(first_color))
        average_f=(first_color[0]+first_color[1]+first_color[2])/3
        #print('average_f: '+str(average_f))
        average_s=0
        counter=0
        while((x!=limit_x2) and (x>0)):
            
            second_color=pix[x,y]
            average_s+=(second_color[0]+second_color[1]+second_color[2])/3
            counter+=1
            # Get the RGBA Value of the a pixel of an image
            #pix[x,y] = (255,255,255)  # Set the RGBA Value of the image (tuple)
            x=x+1*sign
        average_s=average_s/counter
        s.append(average_s)
        #print('average_s: '+str(average_s))
        #flag=abs(limit_x1-flag)
        #print('flag: '+str(flag))
        
        b+=1
    #print(abs(s[0]-s[1]))
    if(abs(s[0]-s[1])>10):
        hump_nose=100
    else:
        hump_nose=(abs(s[0]-s[1]))/0.1
    #im.save(file_name)


    return hump_nose

def forehead(predictor_model,file_name,pose_landmarks):
    smooth=0
    convex=0
    im = Image.open(file_name) # Can be many different formats.
    pix = im.load()
    x1=pose_landmarks.part(18).x
    x2=pose_landmarks.part(25).x
    y1=pose_landmarks.part(18).y
    y2=pose_landmarks.part(25).y
    x=round(x1)
    yn=(2*pose_landmarks.part(18).y-pose_landmarks.part(37).y)
    x_data=[]
    r=[]
    g=[]
    b=[]

    max=0
    min=255
    
    while((x<x2) and (x>0)):
        y=round((((y2-y1)*(x-x1))+yn*(x2-x1))/(x2-x1))
        second_color=pix[x,y]
        r.append(second_color[0])
        g.append(second_color[1])
        b.append(second_color[2])
        x_data.append(x)
        x+=1

    x=round(x1+(x2-x1)*0.25)
    print(x1)
    print(x2)
    print(round(x1+abs(x2-x1)*0.25))
    print(round(x1+abs(x2-x1)*0.75))
    half_x=[]
    half=[]
    while x<round(x1+(x2-x1)*0.75):
        half_x.append(x)
        half.append(-1)
        x+=1

    sum_changes=[]
    i=1
    sum_changes.append(math.sqrt((r[i]-r[i-1])**2+(g[i]-g[i-1])**2+(b[i]-b[i-1])**2))
    sum=0
    while(i<len(r)):
        sum_changes.append(math.sqrt((r[i]-r[i-1])**2+(g[i]-g[i-1])**2+(b[i]-b[i-1])**2))
        sum+=math.sqrt((r[i]-r[i-1])**2+(g[i]-g[i-1])**2+(b[i]-b[i-1])**2)
        i+=1
    aver=sum/len(r)

    start=x1
    end=x2
    i=round(len(sum_changes)*0.25)
    while (i>0):
        if(sum_changes[i]>12):
            start=x1+i
        i-=1
    i=round(len(sum_changes)*0.75)
    while (i<len(sum_changes)):
        if(sum_changes[i]>12):
            end=x1+i
        i+=1

    x=start
    while(x<end):
        
        y=round((((y2-y1)*(x-x1))+yn*(x2-x1))/(x2-x1))
        second_color=pix[x,y]
        average_s=(second_color[0]+second_color[1]+second_color[2])/3
        pix[x,y] = (255,255,255)

        if(average_s<min):
            min=average_s
        if(average_s>max):
            max=average_s
        
        x+=1
    print('max: ',max)
    print('min: ',min)

    dist1=scriptsVector.distance(x1,y1,x2,y2)
    dist2=scriptsVector.distance(start,round((((y2-y1)*(start-x1))+yn*(x2-x1))/(x2-x1)),end,round((((y2-y1)*(end-x1))+yn*(x2-x1))/(x2-x1)))

    if((max-min)>130):
        smooth=0
    elif((max-min)>50 and (max-min)<130):
        smooth=(100-(max-min-40)/0.8)*(dist2/dist1)
    else:
        smooth=100
    
    #fig = plt.figure()
    # Добавление на рисунок прямоугольной (по умолчанию) области рисования
    #graph1 = plt.plot(x_data, r, color='red', label = 'r|max'+str(int(max)))
    #graph1 = plt.plot(x_data, g, color='green', label = u'g|min'+str(int(min)))
    #graph1 = plt.plot(x_data, b, color='blue', label = u'b|dif'+str(int(abs(min-max))))

    #graph1 = plt.plot(x_data, sum_changes, color='black', label = 's_ch')

    #plt.legend()
    #grid1 = plt.grid(True) # линии вспомогательной сетки

    #plt.show()
    #fig.savefig(file_name.replace(".jpg", "_.jpg"))

    convex=100-smooth
    print('convex: ',convex)
    print('smooth: ',smooth)
        
    im.save(file_name.replace(".jpg", "_line.jpg"))
    return smooth,convex

def eyelids(predictor_model,file_name,pose_landmarks):
    inside=0
    center=0
    outside=0
    im = Image.open(file_name) # Can be many different formats.
    pix = im.load()
    distance_1=scriptsVector.distance(pose_landmarks.part(27).x,pose_landmarks.part(27).y,pose_landmarks.part(0).x,pose_landmarks.part(0).y)
    distance_2=scriptsVector.distance(pose_landmarks.part(27).x,pose_landmarks.part(27).y,pose_landmarks.part(16).x,pose_landmarks.part(16).y)

    if(distance_1>distance_2):
        a=36
    else:
        a=42

    counter=0
    flag=[]
    while(counter<4):
        limit_y1=round(pose_landmarks.part(a+counter).y)
        if(a==36):
            limit_y2=round((pose_landmarks.part(a+counter).y)-(pose_landmarks.part(41).y-pose_landmarks.part(37).y))
        else:
            limit_y2=round((pose_landmarks.part(a+counter).y)-(pose_landmarks.part(46).y-pose_landmarks.part(44).y))
        
        x1=pose_landmarks.part(27).x
        x2=pose_landmarks.part(33).x
        y1=pose_landmarks.part(27).y
        y2=pose_landmarks.part(33).y
        y=limit_y1
        y_data=[]
        average_s_data=[]
        while(y>limit_y2):
            color=pix[round(((y-y1)*(x2-x1)+pose_landmarks.part(a+counter).x*(y2-y1))/(y2-y1)),y]
            average_s=(color[0]+color[1]+color[2])/3
            y_data.append(y)
            average_s_data.append(average_s)
            #print(str(average_f))
            pix[pose_landmarks.part(a+counter).x,y] = (255,255,255)
            y-=1
        fig = plt.figure()
        # Добавление на рисунок прямоугольной (по умолчанию) области рисования
        graph1 = plt.plot(y_data, average_s_data, label = str(a+counter))

        plt.legend()
        grid1 = plt.grid(True) # линии вспомогательной сетки

        #plt.show()
        fig.savefig(file_name.replace(".jpg", str(a+counter)+"__.jpg"))

        test_data = average_s_data[0]
        diff=[]
        for data in average_s_data:
            if(test_data>data)and((test_data-data)>5):
                diff.append(1)
                test_data=data
            else:
                diff.append(0)
                test_data=data

        
        if 1 in diff:
            flag.append(1)
        else:
            flag.append(0)
        counter+=1

    if (flag[1]==0 and flag[2]==0):
        center=100
    if (flag[1]==0 and flag[2]==1):
        center=50
    if (flag[1]==1 and flag[2]==0):
        center=50
    if (flag[0]==1 and flag[3]==1):
        inside=100
        outside=100

    if (flag[0]==0 and flag[3]==1 and a==36):
        outside=100
    if (flag[0]==1 and flag[3]==0 and a==36):
        inside=100
    if (flag[0]==0 and flag[3]==1 and a==42):
        inside=100
    if (flag[0]==1 and flag[3]==0 and a==42):
        outside=100
    if (flag[0]==0 and flag[3]==0):
        inside=100
        outside=100
    if (flag[0]==1 and flag[3]==1):
        inside=0
        outside=0
    
    

    im.save(file_name.replace(".jpg", "_line.jpg"))
    return inside, center, outside


def hair_color(predictor_model,file_name,pose_landmarks):
    light=0
    dark=0 
    orange=0
    im = Image.open(file_name) # Can be many different formats.
    pix = im.load()
    limit_y1=round(pose_landmarks.part(27).y)
    limit_y2=round(pose_landmarks.part(27).y-(pose_landmarks.part(8).y-pose_landmarks.part(27).y))
    x1=pose_landmarks.part(27).x
    x2=pose_landmarks.part(33).x
    y1=pose_landmarks.part(27).y
    y2=pose_landmarks.part(33).y
    y=round(pose_landmarks.part(27).y)
    first_color=pix[round(((y-y1)*(x2-x1)+x1*(y2-y1))/(y2-y1)),y]
    #print('first_color: '+str(first_color))
    average_f=(first_color[0]+first_color[1]+first_color[2])/3
    #print('average_f: '+str(average_f)) 
    y_data=[]
    r=[]
    g=[]
    b=[]
    counter=0
    flag = 0 
    
    while(y>limit_y2)and(flag==0):
        pix_x=round(((y-y1)*(x2-x1)+x1*(y2-y1))/(y2-y1))
        second_color=pix[pix_x,y]
        average_s=(second_color[0]+second_color[1]+second_color[2])/3
        
        y_data.append(y)
        r.append(second_color[0])
        g.append(second_color[1])
        b.append(second_color[2])
        if(counter>1):
            r_color=r[counter-1]
            g_color=g[counter-1]
            b_color=b[counter-1]
            color1_rgb = sRGBColor(r_color / 255, g_color / 255, b_color / 255)

            # Convert from RGB to Lab Color Space
            color1_lab = convert_color(color1_rgb, LabColor)
            r_color=r[counter]
            g_color=g[counter]
            b_color=b[counter]
            color2_rgb = sRGBColor(r_color / 255, g_color / 255, b_color / 255)

            # Convert from RGB to Lab Color Space
            color2_lab = convert_color(color2_rgb, LabColor)
            delta_e = delta_e_cie2000(color1_lab, color2_lab)

            if(delta_e > 7):
                flag==1
        # Get the RGBA Value of the a pixel of an image
        #pix[pix_x,y] = (255,255,255)  # Set the RGBA Value of the image (tuple)
        
        counter+=1
        y-=1
    
    light=(0.299*second_color[0] + 0.587*second_color[1] + 0.114*second_color[2])/2.55
    dark=100-light
    orange=(abs(255-second_color[0])+abs(second_color[1]-102)+second_color[2])/6.63

    return light, dark, orange
