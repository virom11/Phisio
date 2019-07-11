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

        # Load all the models we need: a detector to find the faces, a shape predictor
        # to find face landmarks so we can precisely localize the face
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(predictor_path)

        # Load the image using Dlib
        img = dlib.load_rgb_image(face_file_path)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)

        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(face_file_path))

        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(img, detection))

        window = dlib.image_window()

        # Get the aligned face images
        # Optionally: 
        # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
        images = dlib.get_face_chips(img, faces, size=320)
        for image in images:
            window.set_image(image)
            dlib.hit_enter_to_continue()

        # It is also possible to get a single chip
        image = dlib.get_face_chip(img, faces[0])
        window.set_image(image)
        dlib.hit_enter_to_continue()


