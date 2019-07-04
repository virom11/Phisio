import sys
import dlib
import detect
import detectEugene
import os
#import openface
import imageio
from PIL import Image, ImageDraw
from skimage import io
from skimage.feature import hog

priznak = []
for i in range(1, 63):
    priznak.append(0)  # Массив значений признаков



max = 0
min = 100
predictor_model = "D:/shape_predictor_68_face_landmarks.dat" # Модель определения 68 точек на лице

dir="D:/Dropbox/Студенты/Брови/Темные густые"
for filename in os.listdir(dir):   # Цикл по всем фоткам этой папки
    count=0 # Счетчик фоток в папке
    file_name=dir+"/"+filename
    print(file_name)
    if (file_name.endswith("_hog.jpg")==0) and (file_name.endswith("_detect.jpg")==0):  # Работаем только с оригиналом фото, не hog и не распознанное
        count=count+1
        #file_name = 'E:/567.jpg'
        #file_name = 'D:/Dropbox/Студенты/Губы/Уголки губ вниз/331919_parni_iz_seriala_dnevniki_vampira.jpg';

        # Create a HOG face detector using the built-in dlib class
        face_detector = dlib.get_frontal_face_detector()
        image1 = Image.open(file_name) # Здесь откроет фото
        # Load the image into an array
        image = io.imread(file_name) # Здесь фото, как массив
        #win = dlib.image_window() # Если нужно выводить лицо на экран
        hog_list, hog_img = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L1',
                                visualize=True, feature_vector=True) # Генерируем hog изображение
        #imageio.imwrite(dir+"/"+filename.replace(".jpg","_hog.jpg"), hog_img) # Если его нужно сохранить
        #win.set_image(hog_img) # Вывод hog изображения
        #print(face_detector.detection_window_height)
        face_pose_predictor = dlib.shape_predictor(predictor_model) # Модель распознавания лица
        #face_aligner = openface.AlignDlib(predictor_model) # Это для выравнивания - сейчас не нужно, но в будушем может понадобится
        detected_faces = face_detector(image, 1) # Находим лица, что такое "1" - не помню

        if len(detected_faces) == 0:
            print("Лица на фото не обнаружено")


        if len(detected_faces) > 1:
            print("Обнаружено более одного лица")

        # Загрузка лица
        #win.set_image(image)
        #draw = ImageDraw.Draw(image1)
        # Loop through each face we found in the image
        if len(detected_faces) == 1: # Если лицо одно, то продолжаем
            for i, face_rect in enumerate(detected_faces):

                # Вывод контура лица
                #print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

                # Вывод лица на экран прямоугольника, если выводим на экран
                #draw.rectangle(((face_rect.left(),face_rect.top()),(face_rect.right(),face_rect.bottom())))
                #win.add_overlay(face_rect)

                # Распознавание на лице
                pose_landmarks = face_pose_predictor(image, face_rect)

                # Выравнивание лица
                #alignedFace = face_aligner.align(1000, image, face_rect,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                #pose_landmarks1 = face_pose_predictor(alignedFace, dlib.rectangle(0, 0, 1000, 1000))
                #win.set_image(alignedFace)
                #for i in  [50,51,52]:
                #    print(pose_landmarks.part(i))

                # Рисование точек на лице
                #win.add_overlay(pose_landmarks)
                #for i in range(68):
                #    draw.rectangle((((pose_landmarks.part(i).x-1), (pose_landmarks.part(i).y-1)), ((pose_landmarks.part(i).x+1), (pose_landmarks.part(i).y+1))), fill="red")
                    #draw.ellipse(((pose_landmarks.part(i).x,pose_landmarks.part(i).y),(pose_landmarks.part(i).x+5,pose_landmarks.part(i).y)+5), fill=128, outline="red")
                    #win.add_overlay_circle(pose_landmarks.part(i), 2)

            #image1.save(dir+"/"+filename.replace(".jpg","_detect.jpg")) # Сохранение лица с метками точек
            #imageio.imwrite(dir+"/"+filename.replace(".jpg","_detect.jpg"), win)

            prop = pose_landmarks.part(57).y - pose_landmarks.part(27).y # Измеряем размер лица чтобы получить относительные размеры черт лица

            '''priznak[21]=detect.lips_gal(pose_landmarks, prop)
            priznak[22]=100-priznak[21]
            print("Верхняя губа с галочкой: ",priznak[21])
            print("Прямая верхняя губа: ", priznak[22])
            priznak[23]=detect.lips_height(pose_landmarks, face_rect.bottom()-face_rect.top())
            priznak[24] =100-priznak[23]
            print("Толстая верхняя губа: ",priznak[23])
            print("Тонкая верхняя губа: ", priznak[24])
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
            priznak[28] =detect.lips_rot(pose_landmarks)
            priznak[29] = 100-priznak[28]
            print("Узкий рот: ", priznak[28])
            print("Широкий рот: ", priznak[29])

            priznak[20]=detect.eye_posadka(pose_landmarks)
            priznak[12] = 100-priznak[20]
            print("Близко-посаженные глаза: ", priznak[12])
            print("Широко-посаженные глаза: ", priznak[20])

            priznak[16],priznak[17],priznak[18],priznak[19]=detect.eye_color(pose_landmarks, image1)
            print("Голубые глаза: ", priznak[16])
            print("Зеленые глаза: ", priznak[17])
            print("Карие и черные глаза: ", priznak[18])
            print("Серые глаза: ", priznak[19])

            priznak[40] =detect.chin_size(pose_landmarks, prop)
            priznak[43] = 100 - priznak[40]
            print("Большой подбородок: ", priznak[40])
            print("Маленький подбородок: ", priznak[43])
            priznak[42] = detect.chin_form(pose_landmarks, prop)
            priznak[41] = 100 - priznak[42]
            print("Квадратный подбородок: ", priznak[41])
            print("Круглый подбородок: ", priznak[42])

            priznak[8] = detect.eyebrows_accreted(pose_landmarks, image1)
            print("Сросшиеся брови: ", priznak[8])
            if priznak[8]>max: max=priznak[8]
            if priznak[8] < min: min = priznak[8]

            priznak[3], priznak[4], priznak[5] = detectEugene.eyebrows(pose_landmarks, prop)
            print("Бровин Домиком: ", priznak[3], "Бровин Полукругом: ", priznak[4], "Бровин Линией: ", priznak[5])

            priznak[44] = detectEugene.fat_chin(pose_landmarks, image1)
            print("Двойной подбородок: ", priznak[44])

            priznak[6] = detectEugene.eyebrows_rise(pose_landmarks, prop)
            print("Бровь с подъёмом: ", priznak[6])'''

            priznak[7], priznak[9] = detectEugene.eyebrows_bold(pose_landmarks, image1)
            print("Брови тёмные, густые:", priznak[9], "Брови светлые, редкие:", priznak[7])



print("Максимум: ", max)
print("Минимум: ", min)



# Wait until the user hits <enter> to close the window
# dlib.hit_enter_to_continue()