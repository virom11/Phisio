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

def asymmetry(pose, image, scale, predictor_model,file_name,):
    scriptsVector.face_aligner_func(predictor_model,file_name)


    
    return 0, 100