#! /usr/bin/env python 
# -*- coding: utf-8 -*-
#Строки для корректной работы с киррилицей в python. Работают даже в закоментированном состоянии

import sys
import dlib
import detect
import detectEugene
import detectVector
import os
import openface
import imageio
from PIL import Image, ImageDraw
from skimage import io
from skimage.feature import hog
import numpy as np
import math
import xlsxwriter
import detect
import detectEugene
import detectVector
import time

amountsheet=0

priznak = []
for i in range(0, 66):
    priznak.append(0)  # Массив значений признаков

predictor_model = "/home/vector/Documents/shape_predictor_68_face_landmarks.dat"

workbook = xlsxwriter.Workbook('system_test.xlsx')
cell_format = workbook.add_format()
cell_format.set_font_color('red')
worksheet = []

def analyzer(control_string,dir,amountsheet):
    worksheet.append(workbook.add_worksheet())

    worksheet[amountsheet].write(0, 0, "Number of photo in folder")
    worksheet[amountsheet].write(1, 1, "File Name")
    worksheet[amountsheet].write(1, 2,  "Found Error")
    worksheet[amountsheet].write(1, 3,  "Test")
    worksheet[amountsheet].write(1, 4,  "Result")
    counter=1
    row=1
    sum=0
    max=0
    min=100
    above=0
    below=0
    digits=[]
    errors_list_1=[]
    errors_list_2=[]

    for filename in os.listdir(dir):   # Цикл по всем фоткам этой папки
        if(counter<10000):
            if (filename.endswith("_hog.jpg")==0) and (filename.endswith("_detect.jpg")==0) and (filename.endswith("_aligned.jpg")==0) and ((filename in photo_with_errors)==0):  # Работаем только с оригиналом фото, не hog и не распознанное

                prop=0
                pose_landmarks=0
                detected_faces=[]
                file_name=dir+"/"+filename
                print("File name is: " + str(file_name) + ". " + "Number of photo: " + str(counter))
                face_detector = dlib.get_frontal_face_detector()
                image1 = Image.open(file_name) # Здесь откроет фото
                image = io.imread(file_name) # Здесь фото, как массив
                hog_list, hog_img = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L1',
                                        visualize=True, feature_vector=True) # Генерируем hog изображение
                face_pose_predictor = dlib.shape_predictor(predictor_model) # Модель распознавания лица
                detected_faces = face_detector(image, 1) # Находим лица, что такое "1" - не помню

                if len(detected_faces) == 0:
                    #print("Лица на фото не обнаружено")
                    errors_list_1.append(file_name)

                if len(detected_faces) > 1:
                    #print("Обнаружено более одного лица")
                    errors_list_2.append(file_name)

                if len(detected_faces) == 1: # Если лицо одно, то продолжаем
                    for i, face_rect in enumerate(detected_faces):
                        pose_landmarks = face_pose_predictor(image, face_rect)

                    prop = math.sqrt((pose_landmarks.part(57).x - pose_landmarks.part(27).x) ** 2 +
                                    (pose_landmarks.part(57).y - pose_landmarks.part(27).y) ** 2)# Измеряем размер лица чтобы получить относительные размеры черт лица
                    worksheet[amountsheet].write(row, 0, counter)
                    worksheet[amountsheet].write(row, 1, file_name)
                    worksheet[amountsheet].write(row, 2,  "No")
                    main=0
                    if(control_string == "Переносица с впадиной: "):
                        a, main = detectVector.nose(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Прямой нос: "):
                        main, a = detectVector.nose(predictor_model, file_name,pose_landmarks)
                    elif(control_string =="Крылья носа очерчены: "):
                        main = detectVector.nose_wings(predictor_model, file_name,pose_landmarks)
                    #elif(control_string =="Брови Домиком: "):
                    #    main, a, b = detectEugene.eyebrows(pose_landmarks, prop)
                    #elif(control_string =="Брови Полукругом: "):                       Ошибка: неизвестная переменная scale_
                    #    a, main, b = detectEugene.eyebrows(pose_landmarks, prop)
                    #elif(control_string =="Брови Линией: "):
                    #    a, b, main = detectEugene.eyebrows(pose_landmarks, prop)
                    elif(control_string == "Прямой лоб : "):
                        main, a = detectVector.forehead(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Выпуклый лоб : "):
                        a, main = detectVector.forehead(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Нос картошкой: "):
                        main, a, b  = detectVector.nose_size(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Курносый нос: "):
                        a, main, b = detectVector.nose_size(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Кончик носа вниз: "):
                        a, b, main = detectVector.nose_size(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Бровь с подъёмом: "):
                        main = detectEugene.eyebrows_rise(pose_landmarks, prop)
                    elif(control_string == "Раздвоенный подбородок: "):
                        main = detectEugene.fat_chin(pose_landmarks, image1)
                    elif(control_string == "Лоб Узкий: "):
                        a, main = detectVector.nose(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Лоб Широкий: "):
                        main, a = detectVector.nose(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Духовный : "):
                        main, a, b  = detectEugene.worlds(pose_landmarks, image1, prop)
                    elif(control_string == " Материальный: "):
                        a, main, b = detectEugene.worlds(pose_landmarks, image1, prop)
                    elif(control_string == " Семейный: "):
                        a, b, main = detectEugene.worlds(pose_landmarks, image1, prop)
                    elif(control_string == "Волосы лба Полукругом: "):
                        main, a, b  = detectEugene.forhead_form(pose_landmarks, image1, prop)
                    elif(control_string == " Буквой М: "):
                        a, main, b = detectEugene.forhead_form(pose_landmarks, image1, prop)
                    elif(control_string == "Квадратный: "):
                        a, b, main = detectEugene.forhead_form(pose_landmarks, image1, prop)
                    elif(control_string == "Горбинка на носу: "):
                        main = detectVector.hump_nose(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Скулы выше уровня глаз: "):
                        main, a, b  = detectEugene.cheekbones(pose_landmarks, image1, prop)
                    elif(control_string == "Скулы на уровне глаз: "):
                        a, main, b = detectEugene.cheekbones(pose_landmarks, image1, prop)
                    elif(control_string == "Скулы ниже уровня глаз: "):
                        a, b, main = detectEugene.cheekbones(pose_landmarks, image1, prop)
                    elif(control_string == "Брови тёмные, густые:"):
                        main, a = detectEugene.eyebrows_bold(pose_landmarks, image1)
                    elif(control_string == "Брови светлые, редкие: "):
                        a, main = detectEugene.eyebrows_bold(pose_landmarks, image1)
                    elif(control_string == "Прижатые уши: "):
                        main, a = detectEugene.eyebrows_bold(pose_landmarks, image1)
                    elif(control_string == "Квадратная мочка уха: "):
                        a, main = detectEugene.eyebrows_bold(pose_landmarks, image1)
                    elif(control_string == "Мочка уха большая: "):
                        main, a = detectEugene.earlobe_size(pose_landmarks, image1, prop)
                    elif(control_string == "Мочка уха маленькая: "):
                        a, main = detectEugene.earlobe_size(pose_landmarks, image1, prop)
                    elif(control_string == "Верхняя губа с галочкой: "):
                        main = detect.lips_gal(pose_landmarks, prop)
                    elif(control_string == "Прямая верхняя губа: "):
                        main = 100-detect.lips_gal(pose_landmarks, prop)
                    elif(control_string == "Толстая верхняя губа: "):
                        main = detect.lips_height(pose_landmarks, face_rect.bottom()-face_rect.top())
                    elif(control_string == "Тонкая верхняя губа: "):
                        main = 100-detect.lips_height(pose_landmarks, face_rect.bottom()-face_rect.top())
                    elif(control_string == "Узкий рот: "):
                        main = detect.lips_rot(pose_landmarks)
                    elif(control_string == "Широкий рот: "):
                        main = 100-detect.lips_rot(pose_landmarks)
                    elif(control_string == "Близко-посаженные глаза: "):
                        main = 100-detect.eye_posadka(pose_landmarks)
                    elif(control_string == "Широко-посаженные глаза: "):
                        main = detect.eye_posadka(pose_landmarks)
                    elif(control_string == "Голубые глаза: "):
                        main,a,b,c=detect.eye_color(pose_landmarks, image1)
                    elif(control_string == "Зеленые глаза: "):
                        a,main,b,c=detect.eye_color(pose_landmarks, image1)
                    elif(control_string == "Карие и черные глаза: "):
                        a,b,main,c=detect.eye_color(pose_landmarks, image1)
                    elif(control_string == "Серые глаза: "):
                        a,b,c,main=detect.eye_color(pose_landmarks, image1)
                    elif(control_string == "Большой подбородок: "):
                        main = detect.chin_size(pose_landmarks, prop)
                    elif(control_string == "Маленький подбородок: "):
                        main = 100-detect.chin_size(pose_landmarks, prop)
                    elif(control_string == "Квадратный подбородок: "):
                        main = 100-detect.chin_form(pose_landmarks, prop)
                    elif(control_string == "Круглый подбородок: "):
                        main = detect.chin_form(pose_landmarks, prop)
            
                    worksheet[amountsheet].write(row, 3, control_string)
                    if(main != "Error"):
                        sum=sum+main
                        digits.append(main)
                        if(max<main):
                            max=main
                        if(min>main):
                            min=main
                        if(main>50):
                            above+=1
                        if(main<50):
                            below+=1
                        worksheet[amountsheet].write(row, 4,  str(main))
                        counter+=1
                        row+=1
                    #print(counter)
        else:
            break

    for error in errors_list_1:
        worksheet[amountsheet].write(row, 0, counter,cell_format)
        worksheet[amountsheet].write(row, 1, file_name,cell_format)
        worksheet[amountsheet].write(row, 2,  "Faces not found",cell_format)
        counter+=1
        row+=1

    for error in errors_list_2:
        worksheet[amountsheet].write(row, 0, counter,cell_format)
        worksheet[amountsheet].write(row, 1, file_name,cell_format)
        worksheet[amountsheet].write(row, 2,  "Too many faces",cell_format)
        counter+=1
        row+=1
            
    average=sum/counter
    worksheet[amountsheet].write(row, 0,  "Average")
    worksheet[amountsheet].write(row, 1,  average)
    row+=1
    worksheet[amountsheet].write(row, 0,  "MAX")
    worksheet[amountsheet].write(row, 1,  max)
    row+=1
    worksheet[amountsheet].write(row, 0,  "MIN")
    worksheet[amountsheet].write(row, 1,  min)
    row+=1
    worksheet[amountsheet].write(row, 0,  "Above 50")
    worksheet[amountsheet].write(row, 1,  above)
    row+=1
    worksheet[amountsheet].write(row, 0,  "Below 50")
    worksheet[amountsheet].write(row, 1,  below)
    digits=sorted(digits, key=int)
    median=digits[len(digits)//2]
    row+=1
    worksheet[amountsheet].write(row, 0,  "Median")
    worksheet[amountsheet].write(row, 1,  median)
    amountsheet+=1
    return amountsheet

base="/home/vector/Documents/Проект/"
features_list={"Переносица с впадиной: ": "Нос/Переносица с впадиной",
"Прямой нос: ":"Нос/Прямой нос",
"Крылья носа очерчены: ":"Нос/Крылья носа очерчены",
"Брови Домиком: ":"Брови/Домиком",
"Брови Полукругом: ":"Брови/Полукруглые",
"Брови Линией: ":"Брови/Прямые",
"Прямой лоб : ":"Лоб/Прямой лоб",
"Выпуклый лоб : ":"Лоб/Выпуклый лоб",
"Нос картошкой: ":"Нос/Нос картошкой",
"Курносый нос: ":"Нос/Курносый нос",
"Кончик носа вниз: ":"Нос/Кончик носа вниз",
"Бровь с подъёмом: ":"Брови/С подъемом",
"Раздвоенный подбородок: ":"Подбородок/Раздвоенный с вмятиной",
"Лоб Широкий: ":"Лоб/Широкий лоб",
"Лоб Узкий: ":"Лоб/Узкий лоб",
"Духовный : ":"Миры/Духовный",
" Материальный: ":"Миры/Материальный",
" Семейный: ":"Миры/Семейный",
"Волосы лба Полукругом: ":"Лоб/Полукруглой рост волос",
" Буквой М: ":"Лоб/Волосы буквой М",
"Квадратный: ":"Лоб/Квадратный рост волос",
"Горбинка на носу: ":"Нос/Нос с горбинкой",
"Скулы выше уровня глаз: ":"Скулы/Скулы выше уровня глаз",
"Скулы на уровне глаз: ":"Скулы/Скулы на уровне глаз",
"Скулы ниже уровня глаз: ":"Скулы/Скулы ниже уровня глаз",
"Брови тёмные, густые:" :"Брови/Темные густые",
"Брови светлые, редкие:":"Брови/Светлые редкие",
"Прижатые уши: ":"Уши/прижатые Уши",
"Квадратная мочка уха: ":"Уши/Квадратная мочка уха",
"Мочка уха большая: ":"Уши/Большая мочка уха",
"Мочка уха маленькая: ":"Уши/Маленькая мочка уха",
"Верхняя губа с галочкой: ": "Губы/Верхняя губа с галочкой", 
"Прямая верхняя губа: ": "Губы/Прямая верхняя губа", 
"Толстая верхняя губа: ": "Губы/Толстая верхняя губа", 
"Тонкая верхняя губа: " : "Губы/Тонкая верхняя губа", 
"Уголки губ вверх: " : "Губы/Уголки губ вверх", 
"Уголки губ вниз: " : "Губы/Уголки губ вниз", 
"Уголки губ прямо: " : "Губы/Уголки губ прямые", 
"Близко-посаженные глаза: " : "Глаза/Близкая посадка глаз", 
"Широко-посаженные глаза: " : "Глаза/широкая посадка глаз", 
"Голубые глаза: " : "Глаза/Голубые", 
"Зеленые глаза: " : "Глаза/Зеленые", 
"Карие и черные глаза: " : "Глаза/Карие и черные", 
"Серые глаза: " : "Глаза/Серые", 
"Большой подбородок: " : "Подбородок/большой подбородок", 
"Маленький подбородок: " : "Подбородок/Маленький подбородок", 
"Квадратный подбородок: " : "Подбородок/Квадратный", 
"Круглый подбородок: " : "Подбородок/Круглый", 
"Сросшиеся брови: " : "Брови/Сросшиеся"}


photo_with_errors={
"/home/vector/Documents/Проект/Брови/Домиком/1.jpg",
"/home/vector/Documents/6pBSWfFebRU.jpg", 
"/home/vector/Documents/17_449160.jpg", 
"/home/vector/Documents/340x464_0xd42ee42a_10290128931424358864.jpeg", 
"/home/vector/Documents/Проект/Брови/Домиком/36918962_2521241227901897_7814025260801458176_n.jpg", 
"/home/vector/Documents/Проект/Брови/Домиком/JQaJYKXtS_4.jpg", 
"/home/vector/Documents/Проект/Брови/Домиком/nd4sOTfSV3c.jpg", 
"/home/vector/Documents/Проект/Брови/Домиком/OmUdqyA0ws4.jpg", 
"/home/vector/Documents/Проект/Брови/Домиком/orig55646.jpg", 
"/home/vector/Documents/Проект/Брови/Домиком/Безымянный98.jpg",
"/home/vector/Documents/Проект/Брови/С подъемом/0ZN2t5dntmE.jpg", 
"/home/vector/Documents/Проект/Брови/С подъемом/1qfl2CR5sfk.jpg", 
"/home/vector/Documents/Проект/Брови/С подъемом/2M60jdnNJP8.jpg", 
"/home/vector/Documents/Проект/Брови/С подъемом/3dQWpz_RqSg.jpg", 
"/home/vector/Documents/Проект/Брови/С подъемом/HVFmT23QktE.jpg", 
"/home/vector/Documents/Проект/Брови/С подъемом/j9ZWI4shJnk.jpg", 
"/home/vector/Documents/Проект/Брови/С подъемом/onL2ilSIg5w.jpg", 
"/home/vector/Documents/Проект/Брови/С подъемом/PtVEJ_03jwM.jpg", 
"/home/vector/Documents/Проект/Брови/С подъемом/TFA-4boN-2Q.jpg", 
"/home/vector/Documents/Проект/Брови/С подъемом/uLTkZhDauBU.jpg", 
"/home/vector/Documents/Проект/Брови/С подъемом/xWBzBOBWF_8.jpg"
}


start_time=time.time()
control_string="Бровь с подъёмом"
dir=base+"Брови/С подъемом"
amountsheet=analyzer(control_string,dir,amountsheet)
hours=int(time.time()-start_time)//3600
minutes=int(time.time()-start_time)//60
print("Time passed: " + str(hours) + ":" + str(minutes) + ":" + str(int((time.time()-start_time)%3600)))

'''
j=0

print('Number of folders: ' + str(len(features_list)))
start_time=time.time()
for i in features_list:
    hours=int(time.time()-start_time)//3600
    minutes=int(time.time()-start_time)//60
    j+=1
    print('Number of folder: ' + str(j))
    print("Time passed: " + str(hours) + ":" + str(minutes) + ":" + str(int((time.time()-start_time))%3600))
    control_string=i
    dir=base+features_list[i]
    amountsheet=analyzer(control_string,dir,amountsheet)
'''

workbook.close()




'''
                left =detect.left_lips_ugolki(pose_landmarks, prop)
                right = detect.right_lips_ugolki(pose_landmarks, prop)
                d=(left+right)/2
                if d>0:
                    priznak[25]=d
                    priznak[26]=0
                    if d<20: priznak[27]=100-d*5
                if (d<0):
                    priznak[25]=0
                    priznak[26] = d
                    if d<20: priznak[27]=100-d*5
                print("Уголки губ вверх: ", priznak[25])
                print("Уголки губ вниз: ", priznak[26])
                print("Уголки губ прямо: ", priznak[27])

                priznak[8] = detect.eyebrows_accreted(pose_landmarks, image1)
                print("Сросшиеся брови: ", priznak[8])
                if priznak[8]>max: max=priznak[8]
                if priznak[8] < min: min = priznak[8]
'''  