#! /usr/bin/env python 
# -*- coding: utf-8 -*-
#Строки для корректной работы с киррилицей в python. Работают даже в закоментированном состоянии

import sys
import dlib
import os
#import openface
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

def crop(img, cords, directory, filename):
    #cv2.imshow("aligned", img)
    crop_img = img[cords[1]:cords[3], cords[0]:cords[2]]
    #cv2.imshow("cropped", crop_img)
    cv2.waitKey(1)

    try:
        os.stat(directory)
        cv2.imwrite(directory + filename, crop_img)    
    except FileNotFoundError:
        os.mkdir(directory) 
        cv2.imwrite(directory + filename, crop_img)

def pose_landmarks_detect(predictor_model, file_name):
    pose_landmarks = 0
    detected_faces = 0
    try:
        face_detector = dlib.get_frontal_face_detector()
        image1 = Image.open(file_name) 
        image = io.imread(file_name) 
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

def resize_image(input_image_path,
                 output_image_path,
                 size, filename):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    #print('The original image size is {wide} wide x {height} ''high'.format(wide=width, height=height))
 
    resized_image = original_image.resize(size)
    width, height = resized_image.size
    #print('The resized image size is {wide} wide x {height} ''high'.format(wide=width, height=height))
    #resized_image.show()
    try:
        os.stat(output_image_path)
        resized_image.save(output_image_path + filename)
    except FileNotFoundError:
        os.mkdir(output_image_path) 
        resized_image.save(output_image_path + filename)
    
 

predictor_model = "/home/vector/Documents/models/shape_predictor_68_face_landmarks.dat" # Модель определения 68 точек на лице
dir = "/home/vector/Documents/data_bases/Lip_corners"

priznak = []
for i in range(0, 66):
    priznak.append(0)  # Массив значений признаков

start_time=time.time()
counter = 0

for filenames in os.listdir(dir): 

    if(filenames != 'up'):
        
        for filename in os.listdir(dir + "/" + filenames):
            counter += 1
            hours=int(time.time()-start_time)//3600
            minutes=int(time.time()-start_time)//60
            print('Number of photo:' , counter)
            print("Time passed: " + str(hours%60) + ":" + str(minutes%60) + ":" + str(int((time.time()-start_time))%60))
            file_name = dir + '/' + filenames + '/' + filename
            print('File name:', file_name)
            pose_landmarks = pose_landmarks_detect(predictor_model, file_name)

            directory_aligned = "/home/vector/Documents/data_bases/Changed/" + filenames + "_aligned/" 
            pose_landmarks = sv.face_aligner_func(predictor_model, file_name, directory_aligned, filename)

            if(pose_landmarks != 0):
                file_name = "/home/vector/Documents/data_bases/Changed/" + filenames + "_aligned/" + filename
                pose_landmarks = pose_landmarks_detect(predictor_model, file_name)

                mauth_points = []

                if(pose_landmarks != 0):

                    for i in range(48,66):
                        mauth_points.append([pose_landmarks.part(i).x, pose_landmarks.part(i).y])

                    max_x, max_y = np.amax(mauth_points, axis = 0)
                    min_x, min_y = np.amin(mauth_points, axis = 0)

                    max_x += int((max_x - min_x) * 0.1)
                    min_x -= int((max_x - min_x) * 0.1)
                    max_y += int((max_y - min_y) * 0.2)
                    min_y -= int((max_y - min_y) * 0.2)
                    """
                    print('Max_x: ', max_x)
                    print('Min_x: ', min_x)
                    print('Max_y: ', max_y)
                    print('Min_y: ', min_y)
                    """

                    img = cv2.imread(file_name)
                    directory = "/home/vector/Documents/data_bases/Changed/" + filenames + "_cropped/" 
                    crop(img, [min_x, min_y, max_x, max_y], directory, filename)

                    resize_image(input_image_path = "/home/vector/Documents/data_bases/Changed/" + filenames + "_cropped/" + filename, output_image_path = "/home/vector/Documents/data_bases/Changed/" + filenames + "_resize/", size=(400, 200), filename = filename)