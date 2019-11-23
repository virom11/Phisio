#! /usr/bin/env python
# -*- coding: utf-8 -*-

from keras.datasets import mnist #Загружаем базу mnist
from keras.datasets import cifar10 #Загружаем базу cifar10
from keras.datasets import cifar100 #Загружаем базу cifar100
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from sklearn.model_selection import train_test_split
from PIL import Image  # Для отрисовки изображений
import matplotlib.pyplot as plt  # Для отрисовки графиков
import numpy as np  # Библиотека работы с массивами
from tensorflow.keras.preprocessing import image  # Для отрисовки изображений
from tensorflow.keras import utils  # Используем дял to_categoricall
from tensorflow.keras.models import Sequential  # Сеть прямого распространения
from keras.datasets import cifar100  # Загружаем базу cifar100
from keras.datasets import cifar10  # Загружаем базу cifar10
from keras.datasets import mnist  # Загружаем базу mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
import tensorflow as tf
import pandas as pd
import time
import sys
import os

def data_out(i):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    plt.show()


dir = "/home/vector/Documents/data_bases/Changed"

start_time = time.time()

images = [] #массив изображений 
y = [] #массив классов изображений
counter = [0,0]

for filenames in os.listdir(dir): 
    counter[0] += 1
    counter[1] = 0

    for filename in os.listdir(dir + "/" + filenames):
        if(counter[1] < 350):

            if filenames.endswith("_resize") == True:

                file_name = dir + '/' + filenames + '/' + filename
                print('File name:', file_name)
                print('Number of photo: ' + str(counter[0]), end="\r", flush=True)

                img = Image.open(file_name)
                img.load()
                data = np.asarray(img, dtype="int32")
                y.append(filenames.replace('_resize', ''))

                images.append(data)
                counter[1] += 1
                counter[0] += 1

print('Number of photo:', str(counter[0]))

images = np.array(images)

data_out(images)

hours = int(time.time() - start_time) // 3600
minutes = int(time.time() - start_time) // 60

y = np.array(y)

y = np.unique(y, return_inverse=1)[1].reshape(y.shape) 
#преобразование массива классов в массив индексов уникальных элементов

n = 100
#l = np.array([images,y])
#df = pd.DataFrame(l)

x_train, x_test, y_train, y_test = train_test_split(images, y, stratify = y, test_size=0.3)
#X.iloc[x_train] # return dataframe train

y_train = utils.to_categorical(y_train, 3)
print("Time passed: " + str(hours % 60) + ":" + str(minutes % 60) + ":" + str(int((time.time() - start_time) % 60)))

#plt.imshow(Image.fromarray(images[n].astype('uint8')))
#plt.show()

# задаём заранее batch_size для сетей
batch_size = 64

#Создаем последовательную модель
model = Sequential()
#Слой пакетной нормализации
model.add(BatchNormalization(input_shape=(200, 400, 3)))
#Первый сверточный слой
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
#Второй сверточный слой
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
#Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
#Слой регуляризации Dropout
model.add(Dropout(0.25))

#Слой пакетной нормализации
model.add(BatchNormalization())
#Третий сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#Четвертый сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
#Слой регуляризации Dropout
model.add(Dropout(0.25))

#Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
#Полносвязный слой для классификации
model.add(Dense(512, activation='relu'))
#Слой регуляризации Dropout
model.add(Dropout(0.25))
#Выходной полносвязный слой
model.add(Dense(3, activation='softmax'))

#Компилируем сеть
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# ширина и высота изображения
img_size = 25

# генератор изображений
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=5,
    width_shift_range=2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.2
)

datagen.fit(x_train)

# prepare iterator
train_generator = datagen.flow(x_train, y_train, batch_size = batch_size)

validation_generator = datagen.flow(x_train, y_train, batch_size = batch_size, subset='validation')

history = model.fit_generator(
    train_generator,
    steps_per_epoch = batch_size,
    validation_data = validation_generator, 
    validation_steps = batch_size,
    epochs=20,
    verbose=1
)

#Оображаем график точности обучения
plt.plot(history.history['acc'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_acc'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()