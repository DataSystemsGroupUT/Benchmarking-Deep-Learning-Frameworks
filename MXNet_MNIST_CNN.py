#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 18:18:35 2018

@author: nesma
"""
from __future__ import print_function
import numpy as np
import mxnet as mx
import psutil
import time
import datetime
import os
from mxnet import nd, autograd, gluon


mx.random.seed(1)
myDevice= mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()


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
    

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

def loadData():
    train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                          pbatchSize, shuffle=True)
    test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                         pbatchSize, shuffle=False)
    return train_data, test_data

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
        
        


def model():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Dropout(0.25))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(128, activation="relu"))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(pnumClasses))
        #softmax takes 10
        
    return  net



def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(myDevice)
        label = label.as_in_context(myDevice)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]



def RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    train_data, test_data = loadData()
    
    #we didn’t have to include the softmax layer 
    #because MXNet’s has an efficient function that simultaneously computes 
    #the softmax activation and cross-entropy loss
    
    #Get Model
    net = model()
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=myDevice)
    
    #Optimizer settings
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
    
    for e in range(pEpochs+1):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(myDevice)
            label = label.as_in_context(myDevice)
            with autograd.record():
                output = net(data)
                loss = gluon.loss.SoftmaxCrossEntropyLoss()(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            curr_loss = nd.mean(loss).asscalar()
            #moving_loss = (curr_loss if ((i == 0) and (e == 0)) else (1 - 0.01) * moving_loss + 0.01 * curr_loss)
        
        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, curr_loss, train_accuracy, test_accuracy))
        

def runModel(dataset,batchSize=128,numClasses=10,epochs=12,learningRate=0.01,momentum=0.5,weightDecay=1e-6):
    if dataset is "mnist":
        RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar10":
        pass   
    else:
        print("Choose cifar10 or mnist")

def main():
    StartMonitoring()
    runModel("mnist",epochs=3)
    EndMonitoring()
    
  
if __name__ == '__main__':
    main()
