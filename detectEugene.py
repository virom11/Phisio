import math
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
from PIL import Image, ImageDraw
from scriptsEugene import *


# Домиком, Кругом, Линией
def eyebrows(pose, scale):
    scale = 100 / scale

    dir1 = dir_between(pose.part(17).x, pose.part(17).y, pose.part(18).x, pose.part(18).y,
                       pose.part(21).x, pose.part(21).y, pose.part(19).x, pose.part(19).y)

    dir2 = dir_between(pose.part(25).x, pose.part(25).y, pose.part(26).x, pose.part(26).y,
                       pose.part(22).x, pose.part(22).y, pose.part(24).x, pose.part(24).y)

    x_circle, y_circle, radius = rad_circle(pose.part(17).x, pose.part(17).y,
                                            pose.part(19).x, pose.part(19).y,
                                            pose.part(21).x, pose.part(21).y, scale)

    x_circle1, y_circle1, radius1 = rad_circle(pose.part(26).x, pose.part(26).y,
                                               pose.part(24).x, pose.part(24).y,
                                               pose.part(22).x, pose.part(22).y, scale)

    eye_house = ((dir1 + dir2) / 2 - 20) * 1.7
    eye_line = ((radius + radius1) / 2 - 20) * 1.7
    eye_circle = 100 - eye_line

    diff = lined(pose.part(20).x, pose.part(20).y, pose.part(19).x, pose.part(19).y, pose.part(21).x, pose.part(21).y)
    diff = abs(diff * 100 / scale_) / 10
    eye_house *= 1 - diff
    eye_circle *= 1 + diff

    eye_house, eye_circle, eye_line = clamp(eye_house, 0, 100), clamp(eye_circle, 0, 100), clamp(eye_line, 0, 100)

    return eye_house, eye_circle, eye_line


# Подбородок с ямкой
def fat_chin(pose, image):
    hdist = (pose.part(8).x - pose.part(7).x)
    vdist = (pose.part(8).y - pose.part(57).y)

    pose_max = min(pose.part(7).y, pose.part(9).y)

    min_x = round(pose.part(8).x - hdist / 3)
    max_x = round(pose.part(8).x + hdist / 3)
    min_y = round(pose_max - vdist / 3)
    max_y = round(pose_max)

    pit_color = get_color(min_x, max_x, min_y, max_y, image)

    min_x = round(pose.part(8).x - hdist / 3)
    max_x = round(pose.part(8).x + hdist / 3)
    min_y = round(pose_max - vdist * (2 / 3))
    max_y = round(pose_max - vdist * (1 / 3))

    chin_color = get_color(min_x, max_x, min_y, max_y, image)

    return clamp((chin_color - pit_color + 50) / 2.5, 0, 100)


# Брови с подъёмом
def eyebrows_rise(pose, scale):
	scale = 100 / scale

	rise1 = distance(pose.part(36).x, pose.part(36).y, pose.part(17).x, pose.part(17).y )
	near1 = distance(pose.part(39).x, pose.part(39).y, pose.part(21).x, pose.part(21).y )

	rise2 = distance(pose.part(45).x, pose.part(45).y, pose.part(26).x, pose.part(26).y )
	near2 = distance(pose.part(42).x, pose.part(42).y, pose.part(22).x, pose.part(22).y )

	rise = (rise1 + rise2) / 2 * scale / 0.4
	rise += ((rise - (near1 + near2) / 2) * scale - 50) / 4

	return clamp(rise, 0, 100)


# Тёмные густые, Светлые редкие - Брови
def eyebrows_bold(pose, image):
    
    #min_x = min(pose.part(17).x, pose.part(18).x, pose.part(19).x, pose.part(20).x, pose.part(21).x)
    #max_x = max(pose.part(17).x, pose.part(18).x, pose.part(19).x, pose.part(20).x, pose.part(21).x)
    #min_y = min(pose.part(17).y, pose.part(18).y, pose.part(19).y, pose.part(20).y, pose.part(21).y)
    #max_y = max(pose.part(17).y, pose.part(18).y, pose.part(19).y, pose.part(20).y, pose.part(21).y)
    

    min_x = min(pose.part(18).x, pose.part(19).x)
    max_x = max(pose.part(18).x, pose.part(19).x) + 1
    min_y = min(pose.part(18).y, pose.part(19).y)
    max_y = min_y + (pose.part(28).y - pose.part(27).y)

    eyebrows_color1 = get_dominate_color(min_x, max_x, min_y, max_y, image)

    min_x = min(pose.part(24).x, pose.part(25).x)
    max_x = max(pose.part(24).x, pose.part(25).x) + 1
    min_y = min(pose.part(24).y, pose.part(25).y)
    max_y = min_y + (pose.part(28).y - pose.part(27).y)

    eyebrows_color2 = get_dominate_color(min_x, max_x, min_y, max_y, image)

    light_rare = ((eyebrows_color1 + eyebrows_color2) / 2 - 100) / 4
    light_rare = clamp(light_rare, 0, 100)
    bold_often = 100 - light_rare

    return light_rare, bold_often


# Форма лица
def forhead_form(pose, image, scale):
	"""
	dir1 = dir_between(pose.part(2).x, pose.part(2).y, pose.part(4).x, pose.part(4).y, 
										pose.part(14).x, pose.part(14).y, pose.part(12).x, pose.part(12).y)

	dir2 = dir_between(pose.part(5).x, pose.part(5).y, pose.part(7).x, pose.part(7).y, 
										pose.part(11).x, pose.part(11).y, pose.part(9).x, pose.part(9).y)

	if dir2 > 73 or dir1 > 40:
		result = "Огонь"
	else:
		result = "Вода"

	if (dir2 / 2) - (dir2 / 14) > dir1:
		result = "Ветер" 
	"""
	forhead = [0, 0, 0]
	forhead[0], forhead[1], forhead[2] = add_forehead(pose, image, scale)

	if forhead[1].length == 16:
		return "Лоб слишком тёмный, либо неправильный угол", "", ""

	distance = lined(forhead[1].x, forhead[1].y, forhead[0].x, forhead[0].y, forhead[2].x, forhead[2].y) * 100/scale

	fh_M = clamp((distance + 100) / 2, 0, 100)
	fh_circle = clamp(100 - fh_M * (1 + (100 - fh_M) / 100), 0, 100)
	fh_square = 100 - clamp(abs(distance) * 2, 0, 100)

	return fh_circle, fh_M, fh_square

def forhead_height(pose, image, scale):
	forhead = [0, 0, 0]
	forhead[0], forhead[1], forhead[2] = add_forehead(pose, image, scale)

	if forhead[1].length == 16:
		return "Лоб слишком тёмный, либо неправильный угол", "Лоб слишком тёмный, либо неправильный угол"

	height = forhead[1].length
	wide = clamp((height - 50) * 1.67, 0, 100)
	narrow = 100 - wide

	return wide, narrow

"""
from __future__ import print_function
import binascii
import struct
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import math


def eyebrows_accreted_upd(pose_landmarks, im):
    NUM_CLUSTERS = 1
    eyebrows_color = [0, 0, 0]
    nose_color = [0, 0, 0]
    goal_color = [0, 0, 0]

    # <editor-fold desc="eyebrows_color">
    min_x = min(pose_landmarks.part(18).x, pose_landmarks.part(20).x)
    max_x = max(pose_landmarks.part(18).x, pose_landmarks.part(20).x) + 1
    min_y = min(pose_landmarks.part(18).y, pose_landmarks.part(20).y)
    max_y = max(pose_landmarks.part(18).y, pose_landmarks.part(20).y) + 1

    area = (min_x, min_y, max_x, max_y)
    cropped_img = im.crop(area)
    ar = np.asarray(cropped_img)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    peak = codes[0]
    eyebrows_color[0], eyebrows_color[1], eyebrows_color[2] = peak[0], peak[1], peak[2]
    # </editor-fold>

    # <editor-fold desc="nose_color">
    min_x = min(pose_landmarks.part(21).x, pose_landmarks.part(22).x)
    max_x = max(pose_landmarks.part(21).x, pose_landmarks.part(22).x) + 1
    max_y = max(pose_landmarks.part(20).y, pose_landmarks.part(23).y) + 1
    min_y = max_y - (pose_landmarks.part(27).y - max_y)

    area = (min_x, min_y, max_x, max_y)
    cropped_img = im.crop(area)
    ar = np.asarray(cropped_img)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    peak = codes[0]
    nose_color[0], nose_color[1], nose_color[2] = peak[0], peak[1], peak[2]

    # </editor-fold>

    # <editor-fold desc="goal_color">
    min_x = min(pose_landmarks.part(21).x, pose_landmarks.part(22).x)
    max_x = max(pose_landmarks.part(21).x, pose_landmarks.part(22).x) + 1
    min_y = min(pose_landmarks.part(21).y,pose_landmarks.part(22).y)
    max_y = round((pose_landmarks.part(27).y - min_y) / 2) + min_y

    area = (min_x, min_y, max_x, max_y)
    cropped_img = im.crop(area)
    ar = np.asarray(cropped_img)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    peak = codes[0]
    goal_color[0], goal_color[1], goal_color[2] = peak[0], peak[1], peak[2]

    # </editor-fold>

    for i in range(0, 3):
        min_diff = min(eyebrows_color[i], nose_color[i])
        eyebrows_color[i], nose_color[i], goal_color[i] = eyebrows_color[i] - min_diff, nose_color[i] - min_diff, goal_color[i] - min_diff
        max_100 = max(eyebrows_color[i], nose_color[i])
        goal_color[i] = goal_color[i] / max_100 * 100



    return 100 - (goal_color[0] + goal_color[0] + goal_color[0])/3"""