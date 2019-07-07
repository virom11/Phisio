import math
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
from PIL import Image, ImageDraw
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

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


# Форма волос лба
def forhead_form(pose, image, scale):
	
	forhead = [0, 0, 0]
	forhead[0], forhead[1], forhead[2] = add_forehead(pose, image, scale)

	if forhead[1].length == 16:
		return "Лоб слишком тёмный, либо неправильный угол", "", ""

	distance = lined(forhead[1].x, forhead[1].y, forhead[0].x, forhead[0].y, forhead[2].x, forhead[2].y) * 100/scale

	fh_M = clamp((distance + 100) / 2, 0, 100)
	fh_circle = clamp(100 - fh_M * (1 + (100 - fh_M) / 100), 0, 100)
	fh_square = 100 - clamp(abs(distance) * 2, 0, 100)

	return fh_circle, fh_M, fh_square


# Высота лба
def forhead_height(pose, image, scale):
	forhead = [0, 0, 0]
	forhead[0], forhead[1], forhead[2] = add_forehead(pose, image, scale)

	if forhead[1].length == 16:
		return "Лоб слишком тёмный", "Лоб слишком тёмный"

	height = forhead[1].length
	wide = clamp((height - 50) * 1.67, 0, 100)
	narrow = 100 - wide

	return wide, narrow


# Высота бровей
def eyebrows_height(pose, image, scale):

	length1 = eyebrows_height_1(pose, image, scale, 20, 38)
	length2 = eyebrows_height_1(pose, image, scale, 23, 43)

	length = (length1 + length2) / 2
	
	if length in range(10, 17):
		length = clamp(50 * (1 + (length - 13) / 100), 0, 100)
	else:
		length = clamp((length - 5) * 5.8, 0, 100)

	return 100 - length, length

def face_form(pose, image, scale):
	forhead = [0, 0, 0]
	forhead[0], forhead[1], forhead[2] = add_forehead(pose, image, scale)

	dist1 = distance(pose.part(17).x, pose.part(17).y, pose.part(26).x, pose.part(26).y)
	dist2 = distance(pose.part(1).x, pose.part(1).y, pose.part(15).x, pose.part(15).y)
	dist3 = distance(pose.part(4).x, pose.part(4).y, pose.part(12).x, pose.part(12).y)
	dist4 = distance(pose.part(5).x, pose.part(5).y, forhead[1].x, forhead[1].y)

	dist1, dist2, dist3, dist4 = dist1 * 100/scale, dist2 * 100/scale, dist3 * 100/scale, dist4 * 100/scale

	water = ((dist1 + dist2 + dist3)/3 - 130) * 2

	wind = 100 - abs(dist3 - dist1) * 6

	fire = mean_square(100 - (dist1 - 130) * 2.5, (dist1 - dist3 + 30) * 1.67)

	if forhead[1].length != 16:
		water *= dist2 / dist4
		wind *= dist4 / dist2
		fire *= (50 - abs(dist2 - dist4) * 5) / 100 + 1

	return clamp(water, 0, 100), clamp(wind, 0, 100), clamp(fire, 0, 100)

"""
from __future__ import print_function
import binascii
import struct
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import math
"""