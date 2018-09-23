#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 15:33:56 2018

@author: nesma
# =============================================================================
# """

from __future__ import print_function
import keras
from keras.datasets import cifar10,mnist
from keras import Sequential,optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import os
import psutil
import time
import datetime





def initParameters(dataset,batchSize,numClasses,epochs):
    global Dataset    
    global pbatchSize
    global pnumClasses
    global pEpochs
    
    Dataset = dataset
    pbatchSize = batchSize
    pnumClasses = numClasses
    pEpochs = epochs
    
        
    

def loadData():
    (x_train, y_train), (x_test, y_test) = Dataset.load_data()
    global imgRows
    global imgCols
    global inputShape
    imgRows = x_train.shape[1]
    imgCols = x_train.shape[2]

    try:
        imgRGB_Dimensions = x_train.shape[3]
    except Exception:
        imgRGB_Dimensions = 1 #For Gray Scale Images

    print(x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], imgRows, imgCols, imgRGB_Dimensions)
    x_test = x_test.reshape(x_test.shape[0], imgRows, imgCols, imgRGB_Dimensions)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = NormalizeData(x_train, x_test)
    y_train, y_test = CategorizeData(y_train,y_test,pnumClasses)
    inputShape = (imgRows, imgCols, imgRGB_Dimensions)
    return x_train, y_train, x_test, y_test

def NormalizeData(x_train,x_test):
    x_train /= 255
    x_test /= 255
    return x_train, x_test
    
# convert class vectors to binary class matrices
# The result is a vector with a length equal to the number of categories.
def CategorizeData(y_train,y_test,pnumClasses):
    y_train = keras.utils.to_categorical(y_train, pnumClasses)
    y_test = keras.utils.to_categorical(y_test, pnumClasses)
    return y_train, y_test

def StartMonitoring():
        process = psutil.Process(os.getpid())
        print("Before memory_percent", process.memory_percent())
        print(psutil.cpu_percent(percpu=True))
        global start
        start = time.time()
    
    
def EndMonitoring():
        end = time.time()        
        process = psutil.Process(os.getpid())
        print("after memory_percent",process.memory_percent())
        print(psutil.cpu_percent(percpu=True))
        print("Time Elapsed")
        print(str(datetime.timedelta(seconds= end - start)))
        print(end - start)     
        
        
    
def RunMNIST():
        x_train, y_train, x_test, y_test = loadData()
 
        # building model layers with the sequential model
        MNIST_model = Sequential()
        
        MNIST_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=inputShape))
        MNIST_model.add(Conv2D(64, (3, 3), activation='relu'))
        MNIST_model.add(MaxPooling2D(pool_size=(2, 2)))
        MNIST_model.add(Dropout(0.25))
        MNIST_model.add(Flatten())
        
        MNIST_model.add(Dense(128, activation='relu'))
        MNIST_model.add(Dropout(0.5))
        MNIST_model.add(Dense(pnumClasses, activation='softmax'))
        
        MNIST_sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
        MNIST_model.compile(loss='categorical_crossentropy',
                      optimizer=MNIST_sgd,
                      metrics=['accuracy'])
    
        StartMonitoring()
        #Training the model
        MNIST_model.fit(x_train, y_train,
                  batch_size=pbatchSize,
                  epochs=pEpochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        
        EndMonitoring()
        
        #Model performance evaluation
        evaluateModel(MNIST_model,x_test, y_test, verbose=1)
       
        #MNIST_predicted_classes = MNIST_model.predict_classes(MNIST_x_test)



def evaluateModel(model,x_test,y_test):
     pLoss, pAcc = model.evaluate(x_test, y_test, verbose=1)
     print("Test Loss", pLoss)
     print("Test Accuracy", pAcc)

def RunCIFAR10():
        
        x_train, y_train, x_test, y_test = loadData()
 
        CIFAR_model = Sequential()
        
        CIFAR_model.add(Conv2D(32, (3, 3), padding='same',input_shape=inputShape))
        CIFAR_model.add(Activation('relu'))
        CIFAR_model.add(Conv2D(32, (3, 3)))
        CIFAR_model.add(Activation('relu'))
        CIFAR_model.add(MaxPooling2D(pool_size=(2, 2)))
        CIFAR_model.add(Dropout(0.25))
        
        CIFAR_model.add(Conv2D(64, (3, 3), padding='same'))
        CIFAR_model.add(Activation('relu'))
        CIFAR_model.add(Conv2D(64, (3, 3)))
        CIFAR_model.add(Activation('relu'))
        CIFAR_model.add(MaxPooling2D(pool_size=(2, 2)))
        CIFAR_model.add(Dropout(0.25))
        CIFAR_model.add(Flatten())
        CIFAR_model.add(Dense(512))
        CIFAR_model.add(Activation('relu'))
        
        CIFAR_model.add(Dropout(0.5))
        CIFAR_model.add(Dense(pnumClasses))
        CIFAR_model.add(Activation('softmax')) 
        
        CIFAR_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        
        StartMonitoring()
        
        CIFAR_model.fit(x_train, y_train,
                      batch_size=pbatchSize,
                      epochs=pEpochs,
                      validation_data=(x_test, y_test),
                      shuffle=True)
        
        EndMonitoring()
        
        # Score trained model.
        evaluateModel(CIFAR_model,x_test, y_test, verbose=1)
       
        
def runModel(dataset,batchSize=128,numClasses=10,epochs=12):
    initParameters(dataset,batchSize,numClasses,epochs)
    if Dataset is mnist:
        RunMNIST()
    elif Dataset is cifar10:
        RunCIFAR10()     
    else:
        print("Choose cifar10 or mnist")

def main():
    runModel(mnist,epochs=1)
    #runModel(cifar10,epochs=1)
  
if __name__ == '__main__':
        main()
       
