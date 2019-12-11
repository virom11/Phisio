#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Строки для корректной работы с киррилицей в python. Работают даже в закоментированном состоянии
import numpy as np
import sys
import dlib
#from google.colab import files
import matplotlib.pyplot as plt
import os
import openface
#import openface.openface.align_dlib as openface
import imageio
from PIL import Image, ImageDraw
from skimage import io
from skimage.feature import hog
import numpy as np
import math
from sys import platform
import cv2
import time
from datetime import datetime
import traceback

classes = ['Вверх', 'Вниз', 'Прямо']


def data_out(i):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    plt.show()


images = []  # массив изображений
y = []  # массив классов изображений


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


def face_aligner_func(predictor_path, image):
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

    return alignedFace, error


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


# Модель определения 68 точек на лице
predictor_model = "/home/vector/Documents/models/shape_predictor_68_face_landmarks.dat"
dir = '/home/vector/Documents/data_bases/Уголки губ'

errors = []

priznak = []
for i in range(0, 66):
    priznak.append(0)  # Массив значений признаков

start_time = datetime.now()
# первая переменная - счетчик всех обработанных фото, вторая переменная - счетчик обработанных фото в папке
counter = [0, 0]
error = 0

all_photoes = 0
for filenames in os.listdir(dir):
    if(filenames == 'Прямо'):
        for filename in os.listdir(dir + "/" + filenames):
            all_photoes += 1

print('В директории', all_photoes, 'фотографий.')

for filenames in os.listdir(dir):
  # переход по всем папкам директории
    counter[1] = 0
    work_counter = 0
    for filename in os.listdir(dir + "/" + filenames):
        counter[1] += 1
        counter[0] += 1
        work_counter += 1
        if(filenames == 'Прямо'):
            # переход по всем изображениям в папке
            end_time = datetime.now()
            file_name = dir + '/' + filenames + '/' + filename

            print('Number of photo in directory:',
                  counter[0], '. Number of photo in folder:', counter[1], '. File name:', file_name)
            print('Time passed: {}'.format(end_time - start_time))
            print('Time left: {}'.format((end_time - start_time) /
                                         work_counter * (all_photoes - work_counter)))
            try:
                image = io.imread(file_name)
                #plt.imshow(image, cmap=plt.cm.binary)
                # plt.show()
                try:
                    image, error = face_aligner_func(predictor_model, image)
                    #plt.imshow(image, cmap=plt.cm.binary)
                    # plt.show()
                    if(error != 1):
                        pose_landmarks = 0
                        try:
                            pose_landmarks = pose_landmarks_detect(
                                predictor_model, image)

                            mauth_points = []

                            points_list = [53, 54, 55, 64]
                            for i in points_list:
                                mauth_points.append(
                                    [pose_landmarks.part(i).x, pose_landmarks.part(i).y])

                            max_x, max_y = np.amax(mauth_points, axis=0)
                            min_x, min_y = np.amin(mauth_points, axis=0)

                            max_x += int((max_x - min_x) * 0.5)
                            # min_x -= int((max_x - min_x) * 0.3)
                            max_y += int((max_y - min_y) * 0.1)
                            min_y -= int((max_y - min_y) * 0.1)
                            try:
                                image = crop(
                                    image, [min_x, min_y, max_x, max_y])
                                #plt.imshow(image, cmap=plt.cm.binary)
                                #plt.show()
                                # print(np.array(image).shape)
                                try:
                                    image = resize_image(
                                        image, size=(100, 100))
                                    #plt.imshow(image, cmap=plt.cm.binary)
                                    #plt.show()
                                    try:
                                        y.append(int(classes.index(filenames)))
                                        print('Shape of formed y:', np.array(y).shape)
                                        images.append(image)
                                        print('Shape of formed x:', np.array(images).shape)

                                    except:
                                        errors.append([str('Number of photo in directory: ' + str(counter[0]) + '. Number of photo in folder: ' + str(counter[1]) + '. File name:' + str(file_name)),
                                                       'Ошибка записи данных в массив'])
                                        print(errors[-1])
                                        print('Ошибка:\n', traceback.format_exc())
                                        
                                except:
                                    errors.append([str('Number of photo in directory: ' + str(counter[0]) + '. Number of photo in folder: ' + str(counter[1]) + '. File name:' + str(file_name)),
                                                   'Ошибка изменения размера фото'])
                                    print(errors[-1])
                                    print('Ошибка:\n', traceback.format_exc())

                            except:
                                errors.append([str('Number of photo in directory: ' + str(counter[0]) + '. Number of photo in folder: ' + str(counter[1]) + '. File name:' + str(file_name)),
                                               'Ошибка обрезки фото'])
                                print(errors[-1])
                                print('Ошибка:\n', traceback.format_exc())

                        except:
                            errors.append([str('Number of photo in directory: ' + str(counter[0]) + '. Number of photo in folder: ' + str(counter[1]) + '. File name:' + str(file_name)),
                                           'Ошибка определения точек лица на фото'])
                            print(errors[-1])
                            print('Ошибка:\n', traceback.format_exc())

                except:
                    errors.append([str('Number of photo in directory: ' + str(counter[0]) + '. Number of photo in folder: ' + str(counter[1]) + '. File name:' + str(file_name)),
                                   'Ошибка выравнивания фото'])
                    print(errors[-1])
                    print('Ошибка:\n', traceback.format_exc())
            except:
                errors.append([str('Number of photo in directory: ' + str(counter[0]) + '. Number of photo in folder: ' + str(counter[1]) + '. File name:' + str(file_name)),
                               'Ошибка загрузки фото'])
                print(errors[-1])
                print('Ошибка:\n', traceback.format_exc())

            print('Amount of errors:', len(errors))
            print('')



np.save('/home/vector/Documents/models/images2.npy', images)
np.save('/home/vector/Documents/models/y2.npy', y)


'''
from numba import jit, cuda 
import numpy as np 
# to measure exec time 
from timeit import default_timer as timer    
  
# normal function to run on cpu 
def func(a):                                 
    for i in range(10000000): 
        a[i]+= 1      
  
# function optimized to run on gpu  
@jit(target ="cuda")                          
def func2(a): 
    for i in range(10000000): 
        a[i]+= 1
if __name__=="__main__": 
    n = 10000000                            
    a = np.ones(n, dtype = np.float64) 
    b = np.ones(n, dtype = np.float32) 
      
    start = timer() 
    func(a) 
    print("without GPU:", timer()-start)     
      
    start = timer() 
    func2(a) 
    print("with GPU:", timer()-start)


'''
