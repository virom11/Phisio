#! /usr/bin/env python 
# -*- coding: utf-8 -*-
#Строки для корректной работы с киррилицей в python. Работают даже в закоментированном состоянии

import sys
import dlib
import os
import openface
import imageio
import numpy as np
import math
import xlsxwriter
import time
from PIL import Image, ImageDraw
from skimage import io
from skimage.feature import hog

import detect
import detectEugene
import detectVector
from features_list import features_list

priznak = []
for i in range(0, 66):
    priznak.append(0)  # Массив значений признаков

file_num=0
predictor_model = "/home/vector/Documents/shape_predictor_68_face_landmarks.dat"
base="/home/vector/Documents/Проект/"

max = 0
min = 100
f = open('tests/problems_photo')
errors={}
for line in f:
    #print(line)
    adress=""
    result=""
    flag = 0
    if(".jpg" or ".jpeg")in line:
        for symbol in line:
            if(flag>=6):
                result=result+symbol
            else:
                adress=adress+symbol 
            if(symbol=="."):
                flag+=1
            elif((symbol=="j")and(flag>=1)):
                flag+=1
            elif((symbol=="p")and(flag>=1)):
                flag+=1
            elif((symbol=="g")and(flag>=1)):
                flag+=1
            elif((symbol==" ")and(flag>=1)):
                flag+=10
        errors[adress]=result

for error in errors:
    flag=0
    final_result=""
    comment=""
    for symbol in errors[error]:
        if(symbol==" "):
            flag=1
        if(flag!=1):
            if(symbol=="1"):
                final_result+="Неправильное определение точек лица. "
            elif(symbol=="2"):
                final_result+="Поврежденный файл. "
            elif(symbol=="а"):
                final_result+="Лицо закрыто другими объектами. "
            elif(symbol=="б"):
                final_result+="Неправильный ракурс. "
            elif(symbol=="в"):
                final_result+="Маленькое лицо, относительно фото. "
            elif(symbol=="г"):
                final_result+="Неизвестная причина. "
            elif(symbol=="д"):
                final_result+="Черно-белое фото. "   
            elif(symbol=="е"):
                final_result+="Плохое освещение. "
        else:
            comment=comment+symbol
    final_result+=comment
    errors[error]=final_result


def analyzer(control_string,dir,file_num):
    counter=float(1)
    file_num+=1
    row=1
    sum=0
    max=0
    min=100
    above=0
    below=0
    digits=[]
    errors_list_1=[]
    errors_list_2=[]

    name='tests/system_test_'+str(file_num)+'.xlsx'
    workbook = xlsxwriter.Workbook(name)
    cell_format = workbook.add_format()
    cell_format.set_font_color('red')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, "Номер")
    worksheet.write(0, 1, "Имя файла")
    worksheet.write(0, 2,  "Ошибка")
    worksheet.write(0, 3,  "Тест")
    worksheet.write(0, 4,  "Результат")
    
    for filename in os.listdir(dir):   # Цикл по всем фоткам этой папки

        some_str=(dir+"/"+filename).replace("/home/vector/Documents/Проект/", "")
        global_flag=0
        for err in errors:
            if some_str in err:
                global_flag=1
            
        if(counter<10000):

            if (filename.endswith("_hog.jpg")==0) and (filename.endswith("_detect.jpg")==0) and (filename.endswith("_aligned.jpg")==0) and (global_flag==0):  # Работаем только с оригиналом фото, не hog и не распознанное
                prop=0
                pose_landmarks=0
                detected_faces=[]
                file_name=dir+"/"+filename
                print("File name is: " + str(file_name) + ". " + "Number of photo: " + str(int(counter)))
                face_detector = dlib.get_frontal_face_detector()
                image1 = Image.open(file_name) # Здесь откроет фото
                image = io.imread(file_name) # Здесь фото, как массив
                hog_list, hog_img = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L1',
                                        visualize=True, feature_vector=True) # Генерируем hog изображение
                face_pose_predictor = dlib.shape_predictor(predictor_model) # Модель распознавания лица
                detected_faces = face_detector(image, 1) # Находим лица, что такое "1" - не помню

                if len(detected_faces) == 0 and ((file_name in errors_list_1)==0):
                    #print("Лица на фото не обнаружено")
                    errors_list_1.append(file_name)

                if len(detected_faces) > 1 and ((file_name in errors_list_2)==0):
                    #print("Обнаружено более одного лица")
                    errors_list_2.append(file_name)
                

                if len(detected_faces) == 1: # Если лицо одно, то продолжаем
                    for i, face_rect in enumerate(detected_faces):
                        pose_landmarks = face_pose_predictor(image, face_rect)

                    prop = math.sqrt((pose_landmarks.part(57).x - pose_landmarks.part(27).x) ** 2 +
                                    (pose_landmarks.part(57).y - pose_landmarks.part(27).y) ** 2)# Измеряем размер лица чтобы получить относительные размеры черт лица
                    worksheet.write(row, 0, counter)
                    worksheet.write(row, 1, file_name.replace("/home/vector/Documents", ""))
                    worksheet.write(row, 2,  "Нет")
                    main=-2
                    var_25 = -1
                    var_26 = -1
                    var_27 = -1
                    #print('Control_string: '+ str(control_string))
                    if(control_string == "Переносица с впадиной: "):
                        a, main = detectVector.nose(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Прямой нос: "):
                        main, a = detectVector.nose(predictor_model, file_name,pose_landmarks)
                    elif(control_string =="Крылья носа очерчены: "):
                        main = detectVector.nose_wings(predictor_model, file_name,pose_landmarks)
                    elif(control_string =="Брови Домиком: "):
                        main, a, b = detectEugene.eyebrows(pose_landmarks, prop)
                    elif(control_string =="Брови Полукругом: "):                       
                        a, main, b = detectEugene.eyebrows(pose_landmarks, prop)
                    elif(control_string =="Брови Линией: "):
                        a, b, main = detectEugene.eyebrows(pose_landmarks, prop)
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
                        a, main = detectEugene.forhead_height(pose_landmarks, image1, prop)
                    elif(control_string == "Лоб Широкий: "):
                        main, a = detectEugene.forhead_height(pose_landmarks, image1, prop)
                    elif(control_string == "Духовный : "):
                        main, a, b  = detectEugene.worlds(pose_landmarks, image1, prop)
                    elif(control_string == "Материальный: "):
                        a, main, b = detectEugene.worlds(pose_landmarks, image1, prop)
                    elif(control_string == "Семейный: "):
                        a, b, main = detectEugene.worlds(pose_landmarks, image1, prop)
                    elif(control_string == "Волосы лба Полукругом: "):
                        main, a, b  = detectEugene.forhead_form(pose_landmarks, image1, prop)
                    elif(control_string == "Буквой М: "):
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
                        a, main = detectEugene.eyebrows_bold(pose_landmarks, image1)
                    elif(control_string == "Брови светлые, редкие:"):
                        main, a = detectEugene.eyebrows_bold(pose_landmarks, image1)
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
                    elif(control_string == "Уголки губ вверх: "):
                        left =detect.left_lips_ugolki(pose_landmarks, prop)
                        right = detect.right_lips_ugolki(pose_landmarks, prop)
                        d=(left+right)/2
                        if d>0:
                            var_25=d
                            var_26=0
                            if d<20: var_27=100-d*5
                        if (d<0):
                            var_25=0
                            var_26 = d
                            if d<20: var_27=100-d*5
                        main = var_25
                    elif(control_string == "Уголки губ вниз: "):
                        left =detect.left_lips_ugolki(pose_landmarks, prop)
                        right = detect.right_lips_ugolki(pose_landmarks, prop)
                        d=(left+right)/2
                        if d>0:
                            var_25=d
                            var_26=0
                            if d<20: var_27=100-d*5
                        if (d<0):
                            var_25=0
                            var_26 = d
                            if d<20: var_27=100-d*5
                        main = var_26
                    elif(control_string == "Уголки губ прямо: "):
                        left =detect.left_lips_ugolki(pose_landmarks, prop)
                        right = detect.right_lips_ugolki(pose_landmarks, prop)
                        d=(left+right)/2
                        if d>0:
                            var_25=d
                            var_26=0
                            if d<20: var_27=100-d*5
                        if (d<0):
                            var_25=0
                            var_26 = d
                            if d<20: var_27=100-d*5
                        main = var_27
                    elif(control_string == "Сросшиеся брови: "):
                        main = detect.eyebrows_accreted(pose_landmarks, image1)
                    elif(control_string == "Веки закрытые внутри: "):
                        main,a,b = detectVector.eyelids(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Веки закрытые посередине: "):
                        a,main,b = detectVector.eyelids(predictor_model, file_name,pose_landmarks)
                    elif(control_string == "Веки закрытые снаружи: "):
                        a,b,main = detectVector.eyelids(predictor_model, file_name,pose_landmarks)
                        
                    worksheet.write(row, 3, control_string)
                    if(main >=0 ):
                        sum=sum+float(main)
                        average=float(sum)/(counter)
                        digits.append(main)
                        if(max<main):
                            max=main
                        if(min>main):
                            min=main
                        if(main>50):
                            above+=1
                        if(main<50):
                            below+=1
                        worksheet.write(row, 4,  main)
                        counter+=1
                        row+=1
                    elif(main == -1):
                        print('Error! Incorrect photo. CODE_ERROR = -1')
                    elif(main == -2):
                        print('Error! Incorrect control string. CODE_ERROR = -2')

        else:
            break

    for error in errors_list_1:
        worksheet.write(row, 0, counter,cell_format)
        worksheet.write(row, 1, error.replace("/home/vector/Documents", ""),cell_format)
        worksheet.write(row, 2,  "Лицо не найдено",cell_format)
        counter+=1
        row+=1

    for error in errors_list_2:
        worksheet.write(row, 0, counter,cell_format)
        worksheet.write(row, 1, error.replace("/home/vector/Documents", ""),cell_format)
        worksheet.write(row, 2,  "Найдено больше одного лица",cell_format)
        counter+=1
        row+=1
    
    for adress in errors:   
        if(features_list[control_string] in adress):
            worksheet.write(row, 0, counter,cell_format)
            worksheet.write(row, 1, (base+adress).replace("/home/vector/Documents", ""),cell_format)
            worksheet.write(row, 2, errors[adress],cell_format)
            counter+=1
            row+=1
    
    worksheet.write(row, 0,  "Среднее значение")
    worksheet.write(row, 1,  average)
    row+=1
    worksheet.write(row, 0,  "Максимальное значение")
    worksheet.write(row, 1,  max)
    row+=1
    worksheet.write(row, 0,  "Минимальное значение")
    worksheet.write(row, 1,  min)
    row+=1
    worksheet.write(row, 0,  "Количество элементов выше 50")
    worksheet.write(row, 1,  above)
    row+=1
    worksheet.write(row, 0,  "Количество элементов ниже 50")
    worksheet.write(row, 1,  below)
    digits=sorted(digits, key=int)
    median=digits[len(digits)//2]
    row+=1
    worksheet.write(row, 0,  "Медиана")
    worksheet.write(row, 1,  median)

    workbook.close()
    return file_num



j=0
print('Number of analyzed functions: ' + str(len(features_list)))
start_time=time.time()
for i in features_list:
    hours=int(time.time()-start_time)//3600
    minutes=int(time.time()-start_time)//60
    j+=1
    print('Number of folder: ' + str(j))
    print("Time passed: " + str(hours%60) + ":" + str(minutes%60) + ":" + str(int((time.time()-start_time))%60))
    control_string=i
    dir=base+features_list[i]
    file_num=analyzer(control_string,dir,file_num)
