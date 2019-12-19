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
from random import choice
from string import ascii_letters
import shutil
from shutil import copytree, ignore_patterns
import threading
from multiprocessing import Pool

start_time = datetime.now()

# Модель определения 68 точек на лице
predictor_model = "/home/vector/Documents/models/shape_predictor_68_face_landmarks.dat"

dir = '/home/vector/Documents/data_bases/Nose/process1'

def config():
    photoes = []
    classes = os.listdir(dir)
    for folder in os.listdir(dir):
        photoes.append(len(os.listdir(dir + "/" + folder)))

    print('Founded', photoes[0] + photoes[1], 'photoes in directory.')
    print('Defined', len(classes), 'classes:')
    print(classes)

    return photoes, classes


def data_out(i):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    plt.show()


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
    error = 0
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
            1000,
            image,
            face_rect,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        pose_landmarks = face_pose_predictor(alignedFace, face_rect)

    return alignedFace, error


def crop(img, cords):
    # cv2.imshow("aligned", img)
    crop_img = img[cords[1]:cords[3], cords[0]:cords[2]]
    # cv2.imshow("cropped", crop_img)
    cv2.waitKey(1)
    return crop_img


def resize_image(image, size):
    res = cv2.resize(image,
                     dsize=(size[0], size[1]),
                     interpolation=cv2.INTER_CUBIC)
    return res


def data_merge():

    merge = []

    folders = []
    for folder in os.listdir(dir):
        folders.append(folder)

    print('Founded', len(folders), 'folders in directory.')
    print(folders)

    print('Do you whant merge other folders? [y/n]')
    mess = input()
    while (mess == 'y'):

        print(
            'Which folders do you want to merge? Write indexes of previos array per comma.'
        )

        folder_indexes = input()

        temp = []
        for index in folder_indexes.split(','):
            temp.append(folders[int(index)])

        merge.append(temp)
        print('Do you whant merge other folders? [y/n]')
        mess = input()

    for merge_list in merge:
        print(merge_list)

        for folder in merge_list[1:]:
            for filename in os.listdir(dir + "/" + folder):
                try:
                    if (filename in os.listdir(dir + "/" + merge_list[0])):
                        im1 = Image.open(dir + "/" + merge_list[0] + "/" + filename)
                        im2 = Image.open(dir + "/" + folder + "/" + filename)
                        if(im1 == im2):
                            print('Photo', filename, 'already exist in directory')
                        else:
                            im2.save(dir + "/" + merge_list[0] + "/" + ''.join(choice(ascii_letters) for i in range(12)) + filename)
                        os.remove(dir + "/" + folder + "/" + filename)

                    else:
                        shutil.move(dir + "/" + folder + '/' + filename, dir + "/" + merge_list[0])
                        #os.remove(dir + "/" + folder + "/" + filename)
                except:
                    print('Error:\n', traceback.format_exc())
            shutil.rmtree(dir + "/" + folder)

#data_merge()
photoes, classes = config()

def data_separator(amount, classes):    

    lst1 = os.listdir(dir + "/" + classes[0])
    lst2 = os.listdir(dir + "/" + classes[1])
    os.mkdir('/home/vector/Documents/data_bases/Nose')

    dir2 = '/home/vector/Documents/data_bases/Nose'

    for i in range(int(amount/2)):
        os.mkdir('/home/vector/Documents/data_bases/Nose/process' + str(i))
        os.mkdir('/home/vector/Documents/data_bases/Nose/process' + str(i) + '/' + str(classes[0]))
        os.mkdir('/home/vector/Documents/data_bases/Nose/process' + str(i) + '/' + str(classes[1]))

        start = int(len(lst1)*i/(amount/2))
        stop = int(len(lst1)*(i + 1)/(amount/2))
        if(stop>len(lst1)):
            stop = len(lst1)

        lst_temp = lst1[start:stop]
        for f in lst_temp:
            shutil.copyfile(dir + "/" + classes[0] + '/' + f, '/home/vector/Documents/data_bases/Nose/process' + str(i) + '/' + str(classes[0]) + '/' + f)

        start = int(len(lst2)*i/(amount/2))
        stop = int(len(lst2)*(i + 1)/(amount/2))
        if(stop>len(lst2)):
            stop = len(lst2)

        lst_temp = lst2[start:stop]
        for f in lst_temp:
            shutil.copyfile(dir + "/" + classes[1] + '/' + f, '/home/vector/Documents/data_bases/Nose/process' + str(i) + '/' + str(classes[1]) + '/' + f)
def data_creator(work_list):
    images = []  # массив изображений
    y = []  # массив классов изображений

    priznak = []
    for i in range(0, 66):
        priznak.append(0)  # Массив значений признаков

    errors = []

    # первая переменная - счетчик всех обработанных фото, вторая переменная - счетчик обработанных фото в папке
    counter = [0, 0]
    error = 0

    all_photoes = 0

    for wl in work_list:
        all_photoes += photoes[work_list.index(wl)]

    for folder in os.listdir(dir):
    # переход по всем папкам директории
        counter[1] = 0

        if folder in work_list:

            for filename in os.listdir(dir + "/" + folder):
                counter[1] += 1
                counter[0] += 1

                end_time = datetime.now()
                file_name = dir + '/' + folder + '/' + filename

                print('Number of photo in directory:',
                    counter[0], '. Number of photo in folder:', counter[1], '. File name:', file_name)
                print('Time passed: {}'.format(end_time - start_time))
                print('Time left: {}'.format((end_time - start_time) /
                                            counter[0] * (all_photoes - counter[0])))
                                            
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

                                points_list = [28, 33, 31, 35]
                                for i in points_list:
                                    mauth_points.append(
                                        [pose_landmarks.part(i).x, pose_landmarks.part(i).y])

                                max_x, max_y = np.amax(mauth_points, axis=0)
                                min_x, min_y = np.amin(mauth_points, axis=0)

                                max_x += int((max_x - min_x) * 0.3)
                                min_x -= int((max_x - min_x) * 0.3)
                                max_y += int((max_y - min_y) * 0.4)
                                #min_y -= int((max_y - min_y) * 0.3)
                                try:
                                    image = crop(
                                        image, [min_x, min_y, max_x, max_y])
                                    #plt.imshow(image, cmap=plt.cm.binary)
                                    #plt.show()
                                    #print(np.array(image).shape)
                                    try:
                                        #image = resize_image(image, size=(120, 120))
                                        #plt.imshow(image, cmap=plt.cm.binary)
                                        #plt.show()
                                        #cv2.imwrite('/home/vector/Documents/data_bases/Changed/' + filenames + '/' + filename + '.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                                        try:
                                            y.append(int(classes.index(folder)))
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


        np.save('/home/vector/Documents/models/images' + ''.join(choice(ascii_letters) for i in range(20)) + '.npy', images)
        np.save('/home/vector/Documents/models/y' + ''.join(choice(ascii_letters) for i in range(20)) + '.npy', y)
data_creator([classes[0]])
