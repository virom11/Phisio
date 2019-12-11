#!/usr/bin/python
'''
--------------------------------------------------------------------
graf(x_data=[],y_data=[], save = False, title=None, new_filename='_'): Функция для вывода графиков.
x_data=[] - значения x 
y_data=[] - значения y
title - название графика
save - если True - сохранение в директории
new_filename - если save=True подпись к названию файла для сохранения в директории.

distance(x1,y1,x2,y2): Функция для рассчета расстояния между точками
x1,y1,x2,y2 - координаты точек

face_aligner_func(predictor_path,face_file_path): Функция для выравнивания лица и сохранения полученной фотографии
predictor_path - модель лица из 68 точек
face_file_path - адресс фотографии в дирректории

--------------------------------------------------------------------
'''

import sys 
import dlib 
import cv2
import os
import openface
import random
import imageio
from PIL import Image, ImageDraw
from skimage import io
from skimage.feature import hog
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy
import scipy.misc
import scipy.cluster


def face_aligner_func(predictor_path, face_file_path, path_for_save, filename):
    pose_landmarks = 0
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_path)
    face_aligner = openface.AlignDlib(predictor_path)
    image = cv2.imread(face_file_path)

    detected_faces=face_detector(image,1)
    #print('Found {} faces.'.format(len(detected_faces)))

    for i, face_rect in enumerate(detected_faces):
        #print('-Face# {} found at Left: {} Top:{} Right:{} Bottom: {} '.format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
        #pose_landmarks = face_pose_predictor(image,face_rect)
        alignedFace = face_aligner.align(1000, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        pose_landmarks = face_pose_predictor(alignedFace, face_rect)
        if path_for_save != None:
            try:
                os.stat(path_for_save)
                cv2.imwrite(path_for_save + filename, alignedFace)
            except FileNotFoundError:
                os.mkdir(path_for_save) 
                cv2.imwrite(path_for_save + filename, alignedFace)
        

    return pose_landmarks

def face_aligner(predictor_path, image):
    alignedFace = 0
    pose_landmarks = 0
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_path)
    face_aligner = openface.AlignDlib(predictor_path)

    detected_faces = face_detector(image, 1)
    # print('Found {} faces.'.format(len(detected_faces)))

    for i, face_rect in enumerate(detected_faces):
        # print('-Face# {} found at Left: {} Top:{} Right:{} Bottom: {} '.format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
        # pose_landmarks = face_pose_predictor(image,face_rect)
        alignedFace = face_aligner.align(
            1000, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        pose_landmarks = face_pose_predictor(alignedFace, face_rect)

    return alignedFace


def crop(img, cords):
    # cv2.imshow("aligned", img)
    crop_img = img[cords[1]:cords[3], cords[0]:cords[2]]
    # cv2.imshow("cropped", crop_img)
    cv2.waitKey(1)
    return crop_img


def resize_image(image, size):
    res = cv2.resize(image, dsize=(
        size[0], size[1]), interpolation=cv2.INTER_CUBIC)
    return res


def pose_landmarks_detect(predictor_model, image):
    pose_landmarks = 0
    detected_faces = 0
    try:
        face_detector = dlib.get_frontal_face_detector()
        face_pose_predictor = dlib.shape_predictor(predictor_model)
        detected_faces = face_detector(image, 1)

        if len(detected_faces) == 0 or len(detected_faces) > 1:
            print("Лица на фото не обнаружено")
            pose_landmarks = 0

        if len(detected_faces) > 1:
            print("Обнаружено более одного лица")

        if len(detected_faces) == 1:
            pose_landmarks = []

        if len(detected_faces) == 1:  # Если лицо одно, то продолжаем
            for i, face_rect in enumerate(detected_faces):
                pose_landmarks = face_pose_predictor(image, face_rect)

    except RuntimeError:
        pose_landmarks = 0

    return pose_landmarks


def distance(x1,y1,x2,y2):
    dist=math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

    return dist

def range(val1,val2):

    if(val1>100):
        val1=100
        val2=0
    elif(val1<0):
        val1=0
        val2=100

    return val1,val2

def graf(x_data=[],y_data=[], title=None, save = False, file_name='Empty_name', new_filename='_'):
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data)

    #plt.show()

    if(title!=None):
        ax.set_title(title)

    if((save==True) and (file_name!='Empty_name')):
        fig.savefig(file_name.replace(".jpg", str(new_filename)+".jpg"))
        print('Graf was saved')

def image_size_printer(im):
    print('Image Size: '+str(im.size))

def test_line(x,y,x1,y1,x2,y2,x11, y11):
    return (y-y11)*(x2-x1)-(x-x11)*(y2-y1)

def radical(a,b):
    if (a>0 and b<0) or (a<0 and b>0):
        return True
    else:
        return False 