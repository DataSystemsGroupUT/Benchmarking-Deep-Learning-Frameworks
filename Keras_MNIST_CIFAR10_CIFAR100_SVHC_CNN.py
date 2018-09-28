#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 15:33:56 2018

@author: nesma
# =============================================================================
# """

from __future__ import print_function
import keras
from keras.datasets import cifar10,mnist,cifar100
from keras import Sequential,optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import LoggerYN as YN
import numpy as np
import scipy.io as sio
import utilsYN as uYN

def initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    global Dataset    
    global pbatchSize
    global pnumClasses
    global pEpochs
    global pLearningRate
    global pMomentum
    global pWeightDecay
    Dataset = dataset
    pbatchSize = batchSize
    pnumClasses = numClasses
    pEpochs = epochs
    pLearningRate = learningRate
    pMomentum = momentum
    pWeightDecay = weightDecay
    
def NormalizeData(x_train,x_test):
        x_train /= 255
        x_test /= 255
        return x_train, x_test
    
    # convert class vectors to binary class matrices
    # The result is a vector with a length equal to the number of categories.
def CategorizeData(dataset,y_train,y_test,pnumClasses):
    y_train = keras.utils.to_categorical(y_train, pnumClasses)
    y_test = keras.utils.to_categorical(y_test, pnumClasses)
    
    
    return y_train, y_test
    
def loadData():
    global Dataset
    if Dataset == "mnist":
        Dataset = mnist
    elif Dataset == "cifar100":
        Dataset = cifar100
    else:
        Dataset = cifar10
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


def loadDataSVHC(fname = 'C:/Users/Youssef/Downloads/Compressed/%s_32x32.mat',extra=False):
    """Load the SVHN dataset (optionally with extra images)
    Args:
        extra (bool, optional): load extra training data
    Returns:
        Dataset: SVHN data
    """
    def load_mat(fname):
        data = sio.loadmat(fname)
        X = data['X'].transpose(3, 0, 1, 2)
        y = data['y'] % 10  # map label "10" --> "0"
        return X, y

    data = uYN.Dataset()
    data.classes = np.arange(10)

    X, y = load_mat(fname % 'train')
    data.train_images = X
    data.train_labels = y.reshape(-1)

    X, y = load_mat(fname % 'test')
    data.test_images = X
    data.test_labels = y.reshape(-1)

    if extra:
        X, y = load_mat(fname % 'extra')
        data.extra_images = X
        data.extra_labels = y.reshape(-1)
    
    (x_train, y_train), (x_test, y_test)  = (data.train_images,data.train_labels),(data.test_images,data.test_labels)
    
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
    y_train, y_test = CategorizeData(y_train,y_test,10)
    inputShape = (imgRows, imgCols, imgRGB_Dimensions)
    return x_train, y_train, x_test, y_test


def model_MNIST():
    MNIST_model = Sequential()
    MNIST_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=inputShape))
    MNIST_model.add(Conv2D(64, (3, 3), activation='relu'))
    MNIST_model.add(MaxPooling2D(pool_size=(2, 2)))
    MNIST_model.add(Dropout(0.25))
    MNIST_model.add(Flatten())
    MNIST_model.add(Dense(128, activation='relu'))
    MNIST_model.add(Dropout(0.5))
    MNIST_model.add(Dense(pnumClasses, activation='softmax'))     
    return MNIST_model

def model_CIFAR10():
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
    return CIFAR_model

def model_CIFAR100():
    CIFAR_model = Sequential()
    CIFAR_model.add(Conv2D(128, (3, 3), padding='same',input_shape=inputShape))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(Conv2D(128, (3, 3)))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(MaxPooling2D(pool_size=(2, 2)))
    CIFAR_model.add(Dropout(0.1))
    
    CIFAR_model.add(Conv2D(256, (3, 3), padding='same'))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(Conv2D(256, (3, 3)))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(MaxPooling2D(pool_size=(2, 2)))
    CIFAR_model.add(Dropout(0.25))
    
    CIFAR_model.add(Conv2D(512, (3, 3), padding='same'))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(Conv2D(512, (3, 3)))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(MaxPooling2D(pool_size=(2, 2)))
    CIFAR_model.add(Dropout(0.5))
    
    
    CIFAR_model.add(Flatten())
    CIFAR_model.add(Dense(1024))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(Dropout(0.5))
    CIFAR_model.add(Dense(pnumClasses))
    CIFAR_model.add(Activation('softmax'))    
    return CIFAR_model

def model_SVHC():
    model_SVHC = Sequential()
    model_SVHC.add(Conv2D(48, (5, 5), padding='same',input_shape=inputShape))
    model_SVHC.add(Activation('relu'))
    model_SVHC.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=2))
    model_SVHC.add(Dropout(0.2))
    
    model_SVHC.add(Conv2D(64, (5, 5), padding='same'))
    model_SVHC.add(keras.layers.normalization.BatchNormalization())
    model_SVHC.add(Activation('relu'))
    model_SVHC.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=1))
    model_SVHC.add(Dropout(0.2))
    
    model_SVHC.add(Conv2D(128, (5, 5), padding='same'))
    model_SVHC.add(keras.layers.normalization.BatchNormalization())
    model_SVHC.add(Activation('relu'))
    model_SVHC.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=2))
    model_SVHC.add(Dropout(0.2))
    
    
    model_SVHC.add(Conv2D(160, (5, 5), padding='same'))
    model_SVHC.add(keras.layers.normalization.BatchNormalization())
    model_SVHC.add(Activation('relu'))
    model_SVHC.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=1))
    model_SVHC.add(Dropout(0.2))
    
    model_SVHC.add(Conv2D(192, (5, 5), padding='same'))
    model_SVHC.add(keras.layers.normalization.BatchNormalization())
    model_SVHC.add(Activation('relu'))
    model_SVHC.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=2))
    model_SVHC.add(Dropout(0.2))
    
    model_SVHC.add(Conv2D(192, (5, 5), padding='same'))
    model_SVHC.add(keras.layers.normalization.BatchNormalization())
    model_SVHC.add(Activation('relu'))
    model_SVHC.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=1))
    model_SVHC.add(Dropout(0.2))
    
    model_SVHC.add(Conv2D(192, (5, 5), padding='same'))
    model_SVHC.add(keras.layers.normalization.BatchNormalization())
    model_SVHC.add(Activation('relu'))
    model_SVHC.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=2))
    model_SVHC.add(Dropout(0.2))
    
    
    model_SVHC.add(Conv2D(192, (5, 5), padding='same'))
    model_SVHC.add(keras.layers.normalization.BatchNormalization())
    model_SVHC.add(Activation('relu'))
    model_SVHC.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=1))
    model_SVHC.add(Dropout(0.2))
    
    model_SVHC.add(Flatten())
    
    model_SVHC.add(Dense(3072))
    model_SVHC.add(Activation('relu'))
    
    
    model_SVHC.add(Dense(3072))
    model_SVHC.add(Activation('relu'))
    
    model_SVHC.add(Dropout(0.5))
    model_SVHC.add(Dense(pnumClasses))
    model_SVHC.add(Activation('softmax'))    
     
    return model_SVHC        

def evaluateModel(model,x_test,y_test,verbose):
     pLoss, pAcc = model.evaluate(x_test, y_test, verbose)
     print("Test Loss", pLoss)
     print("Test Accuracy", pAcc)
     
def RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    x_train, y_train, x_test, y_test = loadData()
    MNIST_model = model_MNIST()
    MNIST_sgd = optimizers.SGD(lr=learningRate, decay=weightDecay, momentum=momentum, nesterov=True)
    MNIST_model.compile(loss='categorical_crossentropy',optimizer=MNIST_sgd, metrics=['accuracy'])
    #Training the model
    memT,cpuT = YN.StartLogger("Keras","MNIST")
    MNIST_model.fit(x_train, y_train,batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    #Model performance evaluation
    YN.EndLogger(memT,cpuT)
    evaluateModel(MNIST_model,x_test, y_test, verbose=1)
    #MNIST_predicted_classes = MNIST_model.predict_classes(MNIST_x_test)

def RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    x_train, y_train, x_test, y_test = loadData()
    CIFAR_model = model_CIFAR10()
    CIFAR_sgd = optimizers.SGD(lr=learningRate, decay=weightDecay, momentum=momentum, nesterov=True)
    CIFAR_model.compile(loss='categorical_crossentropy',optimizer=CIFAR_sgd, metrics=['accuracy'])
    memT,cpuT = YN.StartLogger("Keras","CIFAR10")
    CIFAR_model.fit(x_train, y_train,batch_size=batchSize,epochs=epochs,validation_data=(x_test, y_test),shuffle=True)
    YN.EndLogger(memT,cpuT)
    # Score trained model.
    evaluateModel(CIFAR_model,x_test, y_test, verbose=1)

def RunCIFAR100(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    x_train, y_train, x_test, y_test = loadData()
    CIFAR_model = model_CIFAR100()
    CIFAR_sgd = optimizers.SGD(lr=learningRate, decay=weightDecay, momentum=momentum, nesterov=True)
    CIFAR_model.compile(loss='categorical_crossentropy',optimizer=CIFAR_sgd, metrics=['accuracy'])
    memT,cpuT = YN.StartLogger("Keras","CIFAR100")
    CIFAR_model.fit(x_train, y_train,batch_size=batchSize,epochs=epochs,validation_data=(x_test, y_test),shuffle=True)
    YN.EndLogger(memT,cpuT)
    # Score trained model.
    evaluateModel(CIFAR_model,x_test, y_test, verbose=1)

def RunSVHC(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay,fname):
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    x_train, y_train, x_test, y_test = loadDataSVHC(fname)
    SVHC_model = model_SVHC()
    SVHC_sgd = optimizers.SGD(lr=learningRate, decay=weightDecay, momentum=momentum, nesterov=True)
    SVHC_model.compile(loss='categorical_crossentropy',optimizer=SVHC_sgd, metrics=['accuracy'])
    memT,cpuT = YN.StartLogger("Keras","SVHC")
    SVHC_model.fit(x_train, y_train,batch_size=batchSize,epochs=epochs,validation_data=(x_test, y_test),shuffle=True)
    YN.EndLogger(memT,cpuT)
    # Score trained model.
    evaluateModel(SVHC_model,x_test, y_test, verbose=1)       
        
def runModel(dataset,batchSize=128,numClasses=10,epochs=12,learningRate=0.01,momentum=0.5,weightDecay=1e-6):
    
    if dataset is "mnist":
        RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar10":
        RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar100":
        RunCIFAR100(dataset,batchSize,numClasses=100,epochs,learningRate,momentum,weightDecay)
    elif dataset is "SVHC":
        fname = "./%s_32x32.mat"
        RunSVHC(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay,fname)
    else:
        print("Choose cifar10 or mnist or svhc")

def main():
    
    
    runModel("mnist",epochs=3)
    #runModel("cifar10",epochs=3)
    #runModel("SVHC",epochs=1)
  
if __name__ == '__main__':
        main()
