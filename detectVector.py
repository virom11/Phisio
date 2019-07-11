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


predictor_path = "/home/vector/Documents/shape_predictor_68_face_landmarks.dat"
dir="/home/vector/Documents/Скулы на уровне глаз"
for filename in os.listdir(dir):
    face_file_path =dir+"/"+filename
    if (face_file_path.endswith("_hog.jpg")==0) and (face_file_path.endswith("_detect.jpg")==0):
        print("File name is: " + face_file_path)
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(predictor_path)
        img = dlib.load_rgb_image(face_file_path)
        dets = detector(img, 1)

        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(face_file_path))

        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(img, detection))

        window = dlib.image_window()

        images = dlib.get_face_chips(img, faces, size=320)
        for image in images:
            window.set_image(image)
            dlib.hit_enter_to_continue()

        image = dlib.get_face_chip(img, faces[0])
        window.set_image(image)
        dlib.hit_enter_to_continue()

'''

import sys 
import dlib 
import cv2
import os
import openface
predictor_model = "/home/vector/Documents/shape_predictor_68_face_landmarks.dat"

face_detector=dlib.get_frontal_face_detector()
face_pose_predictor=dlib.shape_predictor(predictor_model)
face_aligner=openface.AlignDlib(predictor_model)

dir="/home/vector/Documents/Скулы на уровне глаз"
for face_file_path in os.listdir(dir):
    face_file_path =dir+"/"+face_file_path
    if (face_file_path.endswith("_hog.jpg")==0) and (face_file_path.endswith("_detect.jpg")==0):
        print("File name is: " + face_file_path)
        image = cv2.imread(face_file_path)

        detected_faces=face_detector(image,1)
        print('Found {} faces.'.format(len(detected_faces)))

        for i, face_rect in enumerate(detected_faces):
            print('-Face# {} found at Left: {} Top:{} Right:{} Bottom: {} '.format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
            pose_landmarks=face_pose_predictor(image,face_rect)
            alignedFace=face_aligner.align(534,image,face_rect,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            cv2.imwrite('aligned_face_{}.jpg'.format(i),alignedFace)
'''