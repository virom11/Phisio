#! /usr/bin/env python 
# -*- coding: utf-8 -*-
#Строки для корректной работы с киррилицей в python. Работают даже в закоментированном состоянии

import sys
import dlib
import os
import openface
import imageio
from PIL import Image, ImageDraw
from skimage import io
from skimage.feature import hog
import numpy as np
import math
from sys import platform
import cv2
import scriptsVector as sv
import time
import scipy
    
predictor_model = "/home/vector/Documents/models/shape_predictor_68_face_landmarks.dat" # Модель определения 68 точек на лице

def face_aligner_func_without_save(predictor_path, face_file_path):
    pose_landmarks = 0
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_path)
    face_aligner = openface.AlignDlib(predictor_path)
    image = cv2.imread(face_file_path)

    detected_faces = face_detector(image,1)
    #print('Found {} faces.'.format(len(detected_faces)))

    for i, face_rect in enumerate(detected_faces):
        #print('-Face# {} found at Left: {} Top:{} Right:{} Bottom: {} '.format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
        #pose_landmarks = face_pose_predictor(image,face_rect)
        alignedFace = face_aligner.align(1000, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        pose_landmarks = face_pose_predictor(alignedFace, face_rect)
        if(face_file_path.endswith('.png')):
            cv2.imwrite(face_file_path.replace('.png', '_.png'), alignedFace)
            alignedFace = io.imread(face_file_path.replace('.png', '_.png'))
            os.remove(face_file_path.replace('.png', '_.png'))
        elif(face_file_path.endswith('.jpg')):
            cv2.imwrite(face_file_path.replace('.jpg', '_.jpg'), alignedFace)
            alignedFace = io.imread(face_file_path.replace('.jpg', '_.jpg'))
            os.remove(face_file_path.replace('.jpg', '_.jpg'))
        elif(face_file_path.endswith('.jpeg')):
            cv2.imwrite(face_file_path.replace('.jpeg', '_.jpeg'), alignedFace)
            alignedFace = io.imread(face_file_path.replace('.jpeg', '_.jpeg'))
            os.remove(face_file_path.replace('.jpeg', '_.jpeg'))

        return alignedFace, pose_landmarks

def crop(img, cords):
    #cv2.imshow("aligned", img)
    crop_img = img[cords[1]:cords[3], cords[0]:cords[2]]
    #cv2.imshow("cropped", crop_img)
    return crop_img

def pose_landmarks_detect_withou_save(predictor_model, img):
    pose_landmarks = 0
    detected_faces = 0
    try:
        face_detector = dlib.get_frontal_face_detector()
        image1 = img
        image = img
        face_pose_predictor = dlib.shape_predictor(predictor_model)
        detected_faces = face_detector(image, 1) 

        if len(detected_faces) == 0 or len(detected_faces) > 1:
            print("Лица на фото не обнаружено")
            pose_landmarks = 0

        if len(detected_faces) > 1:
            print("Обнаружено более одного лица")

        if len(detected_faces) == 1:
            pose_landmarks = []

        if len(detected_faces) == 1: # Если лицо одно, то продолжаем
            for i, face_rect in enumerate(detected_faces):
                pose_landmarks = face_pose_predictor(image, face_rect)

    except RuntimeError:
        pose_landmarks = 0

    return pose_landmarks

file_name = "/home/vector/Documents/data_bases/Lip_corners/down/0b240ba39886a70a291a9b45c9ee6292.jpg"
alignedFace, pose_landmarks = face_aligner_func_without_save(predictor_model, file_name)

pose_landmarks = pose_landmarks_detect_withou_save(predictor_model, alignedFace)
print(pose_landmarks)

if(pose_landmarks != 0):
    mouth_points = []

    for i in range(48,66):
        mouth_points.append([pose_landmarks.part(i).x, pose_landmarks.part(i).y])

    max_x, max_y = np.amax(mouth_points, axis = 0)
    min_x, min_y = np.amin(mouth_points, axis = 0)

    max_x += int((max_x - min_x) * 0.1)
    min_x -= int((max_x - min_x) * 0.1)
    max_y += int((max_y - min_y) * 0.2)
    min_y -= int((max_y - min_y) * 0.2)

    img = crop(alignedFace, [min_x, min_y, max_x, max_y])

    #print(img)
    #resize_image_without_save(input_image = img, output_image_path = '/home/vector/Desktop', size=(400, 200), filename = '0b240ba39886a70a291a9b45c9ee6292.jpg')

    res = cv2.resize(img, dsize=(400, 200), interpolation=cv2.INTER_CUBIC)
    im = Image.fromarray(res)

    if(file_name.endswith('.png')):
        im.save(file_name.replace('.png', '_lips.png'))
    elif(file_name.endswith('.jpg')):
        im.save(file_name.replace('.jpg', '_lips.jpg'))
    elif(file_name.endswith('.jpeg')):
        im.save(file_name.replace('.jpeg', '_lips.jpeg'))