#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 16:29:39 2018

@author: nesma
"""

from __future__ import print_function
#Import Dataset, neural network
import keras
from keras.datasets import mnist
from keras.models import Sequential, optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import psutil
import time
import datetime
import numpy as np


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

#Load the dataset
#Split the dataset into train set and test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')



input_shape = (img_rows, img_cols, 1)

# normalizing the data to help with the training
x_train /= 255
x_test /= 255
# convert class vectors to binary class matrices
# The result is a vector with a length equal to the number of categories.

#
#
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




# building model layers with the sequential model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


#Training the model
process = psutil.Process(os.getpid())
print("before memory_percent",process.memory_percent())
print(psutil.cpu_percent(percpu=True))

start = time.time()
print("start",start)
CNNModel = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
end = time.time()

process = psutil.Process(os.getpid())
print("after memory_percent",process.memory_percent())
print(psutil.cpu_percent(percpu=True))


print("end", end)

print("Time Elapsed")
print(str(datetime.timedelta(seconds= end - start)))
print(end - start)


#Model performance evaluation
Test_loss , acc = model.evaluate(x_test, y_test, verbose=1)
print("Test Loss", Test_loss)
print("Test Accuracy", acc)

predicted_classes = model.predict_classes(x_test)
 
 # see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")
 

