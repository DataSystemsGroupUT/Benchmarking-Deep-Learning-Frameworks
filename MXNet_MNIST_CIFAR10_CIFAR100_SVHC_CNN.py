#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 18:18:35 2018
@author: nesma
"""
from __future__ import print_function
import numpy as np
import mxnet as mx
import LoggerYN as YN
import scipy.io as sio
import utilsYN as uYN
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

def NormalizeData(x_train,x_test):
        x_train /= 255
        x_test /= 255
        return x_train, x_test
    
    # convert class vectors to binary class matrices
    # The result is a vector with a length equal to the number of categories.
#def CategorizeData(y_train,y_test,pnumClasses):
#    print(y_train)
#    print(y_train.shape)    
#    b = np.zeros((y_train.shape[0],pnumClasses))
#    b[np.arange(y_train.shape[0]),y_train] = 1
#    y_train = b
#    print(y_train)
#    
#    b = np.zeros((y_test.shape[0],pnumClasses))
#    b[np.arange(y_test.shape[0]),y_test] = 1
#    y_test = b
#    
#    print(y_train)
#    print(y_test)
#    return y_train, y_test 

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

def loadData():
    if(Dataset == "mnist"):
        train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),pbatchSize, shuffle=True)
        test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),pbatchSize, shuffle=False)
        return train_data, test_data
    elif(Dataset ==  "cifar10"):
        train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=True, transform=transform),pbatchSize, shuffle=True)
        test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=False, transform=transform),pbatchSize, shuffle=False)
        return train_data, test_data
    elif(Dataset ==  "cifar100"):
        train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR100(train=True, transform=transform),pbatchSize, shuffle=True)
        test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR100(train=False, transform=transform),pbatchSize, shuffle=False)
        return train_data, test_data
    else:
        pass

def loadDataSVHN(fname,extra=False):
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

    
    x_train = x_train.reshape(x_train.shape[0], imgRows, imgCols, imgRGB_Dimensions)
    x_test = x_test.reshape(x_test.shape[0], imgRows, imgCols, imgRGB_Dimensions)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = NormalizeData(x_train, x_test)
    inputShape = (imgRows, imgCols, imgRGB_Dimensions)
    
    
    x_train = x_train.reshape(len(y_train), imgRows, imgCols, imgRGB_Dimensions)
    x_train = nd.array(x_train, dtype=x_train.dtype)
    label = nd.array(y_train,dtype=y_train.dtype)
    
    x_test = x_test.reshape(len(y_test), imgRows, imgCols, imgRGB_Dimensions)
    x_test = nd.array(x_test, dtype=x_test.dtype)
    labelTest = nd.array(y_test,dtype=y_test.dtype)
    return x_train,label, x_test,labelTest


def modelMNIST():
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
    return net

def modelCIFAR10():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Dropout(0.25))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(0.25))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(pnumClasses))
        #softmax takes 10
    return  net

def modelCIFAR100():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3,strides=1,padding=1, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3,padding=1, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(0.1))
        
        net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(0.25))
        
        
        net.add(gluon.nn.Conv2D(channels=512, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=512, kernel_size=3, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(0.5))
        
        
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(1024, activation="relu"))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(pnumClasses * 10))
        #softmax takes 10
    return  net

def model_SVHN():
    
    model_SVHN = gluon.nn.Sequential()
    with model_SVHN.name_scope():
        model_SVHN.add(gluon.nn.Conv2D(channels=48, kernel_size=5,strides=1,padding=2, activation='relu'))
        model_SVHN.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        model_SVHN.add(gluon.nn.Dropout(0.2))
   
    
        model_SVHN.add(gluon.nn.Conv2D(channels=64, kernel_size=5,strides=1,padding=2))
        model_SVHN.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        model_SVHN.add(gluon.nn.Activation(activation='relu'))
        model_SVHN.add(gluon.nn.MaxPool2D(pool_size=2, strides=1))
        model_SVHN.add(gluon.nn.Dropout(0.2))
        
        model_SVHN.add(gluon.nn.Conv2D(channels=128, kernel_size=5,strides=1,padding=2))
        model_SVHN.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        model_SVHN.add(gluon.nn.Activation(activation='relu'))
        model_SVHN.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        model_SVHN.add(gluon.nn.Dropout(0.2))
            
        model_SVHN.add(gluon.nn.Conv2D(channels=160, kernel_size=5,strides=1,padding=2))
        model_SVHN.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        model_SVHN.add(gluon.nn.Activation(activation='relu'))
        model_SVHN.add(gluon.nn.MaxPool2D(pool_size=2, strides=1))
        model_SVHN.add(gluon.nn.Dropout(0.2))
        
        model_SVHN.add(gluon.nn.Conv2D(channels=192, kernel_size=5,strides=1,padding=2))
        model_SVHN.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        model_SVHN.add(gluon.nn.Activation(activation='relu'))
        model_SVHN.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        model_SVHN.add(gluon.nn.Dropout(0.2))
        
        model_SVHN.add(gluon.nn.Conv2D(channels=192, kernel_size=5,strides=1,padding=2))
        model_SVHN.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        model_SVHN.add(gluon.nn.Activation(activation='relu'))
        model_SVHN.add(gluon.nn.MaxPool2D(pool_size=2, strides=1))
        model_SVHN.add(gluon.nn.Dropout(0.2))
        
        model_SVHN.add(gluon.nn.Conv2D(channels=192, kernel_size=3,strides=1,padding=2))
        model_SVHN.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        model_SVHN.add(gluon.nn.Activation(activation='relu'))
        model_SVHN.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        model_SVHN.add(gluon.nn.Dropout(0.2))
        
        model_SVHN.add(gluon.nn.Conv2D(channels=192, kernel_size=5,strides=1,padding=2))
        model_SVHN.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        model_SVHN.add(gluon.nn.Activation(activation='relu'))
        model_SVHN.add(gluon.nn.MaxPool2D(pool_size=2, strides=1))
        model_SVHN.add(gluon.nn.Dropout(0.2))
        
        model_SVHN.add(gluon.nn.Flatten())
        model_SVHN.add(gluon.nn.Dense(3072, activation="relu"))
        model_SVHN.add(gluon.nn.Dense(3072, activation="relu"))
      
        model_SVHN.add(gluon.nn.Dense(pnumClasses))
    return model_SVHN        

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(myDevice)
        label = label.as_in_context(myDevice)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

def evaluate_accuracySVHN(X,Y,batchSize, net):
    acc = mx.metric.Accuracy()
    i =0 
    print(X.shape,Y.shape,batchSize)
    for data, label in zip(batch(X, batchSize),batch(Y, batchSize)):
        data = data.as_in_context(myDevice)
        label = label.as_in_context(myDevice)
        data = np.transpose(data,(0,3,1,2))
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
        i=i+1
    return acc.get()[1]
        
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    train_data, test_data = loadData()
    
    print(type(train_data))
    #we didn’t have to include the softmax layer 
    #because MXNet’s has an efficient function that simultaneously computes 
    #the softmax activation and cross-entropy loss
    
    #Get Model
    net = modelMNIST()
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=myDevice)
    
    #Optimizer settings
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learningRate, 'momentum': momentum,'wd': weightDecay})
    
    print("MNIST Training Started.....")
    memT,cpuT = YN.StartLogger("MXNET","MNIST")
    for e in range(pEpochs):
        for i, (data, label) in list(enumerate(train_data)):
            data = data.as_in_context(myDevice)
            label = label.as_in_context(myDevice)
            with autograd.record():
                output = net(data)
                loss = gluon.loss.SoftmaxCrossEntropyLoss()(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            curr_loss = nd.mean(loss).asscalar()
        print("Calculating metrics.....")
        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, curr_loss, train_accuracy, test_accuracy))
    print("MNIST Training Finished.....")
    memT,cpuT = YN.StartLogger("MXNET","MNIST")
        
def RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    train_data, test_data = loadData()
    
    #we didn’t have to include the softmax layer 
    #because MXNet’s has an efficient function that simultaneously computes 
    #the softmax activation and cross-entropy loss
    
    #Get Model
    net = modelCIFAR10()
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=myDevice)
    
    #Optimizer settings
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learningRate, 'momentum': momentum,'wd': weightDecay})
    print("CIFAR10 Training Started.....")
    memT,cpuT = YN.StartLogger("MXNET","CIFAR10")
    for e in range(pEpochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(myDevice)
            label = label.as_in_context(myDevice)
            with autograd.record():
                output = net(data)
                loss = gluon.loss.SoftmaxCrossEntropyLoss()(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            curr_loss = nd.mean(loss).asscalar()
        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e+1, curr_loss, train_accuracy, test_accuracy))
    print("CIFAR Training Finished.....")
    memT,cpuT = YN.StartLogger("MXNET","CIFAR10")


def RunCIFAR100(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    train_data, test_data = loadData()
    
    #we didn’t have to include the softmax layer 
    #because MXNet’s has an efficient function that simultaneously computes 
    #the softmax activation and cross-entropy loss
    
    #Get Model
    net = modelCIFAR100()
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=myDevice)
    
    #Optimizer settings
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learningRate, 'momentum': momentum,'wd': weightDecay})
    print("CIFAR100 Training Started.....")
    memT,cpuT = YN.StartLogger("MXNET","CIFAR100")
    for e in range(pEpochs):
        for i, (data, label) in enumerate(train_data):
            print("\r........ " + str(i) + " ........" )
            data = data.as_in_context(myDevice)
            label = label.as_in_context(myDevice)
            with autograd.record():
                output = net(data)
                loss = gluon.loss.SoftmaxCrossEntropyLoss()(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            curr_loss = nd.mean(loss).asscalar()
        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e+1, curr_loss, train_accuracy, test_accuracy))
    print("CIFAR Training Finished.....")
    memT,cpuT = YN.StartLogger("MXNET","CIFAR100")

        
def RunSVHN(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay,fname):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    train_data_x,train_data_y, test_data_x,test_data_y = loadDataSVHN(fname)
   
    #Get Model
    net = model_SVHN()
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=myDevice)
    
    #Optimizer settings
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learningRate, 'momentum': momentum,'wd': weightDecay})
    

    print("SVHN Training Started.....")
    memT,cpuT = YN.StartLogger("MXNET","SVHN")
    for e in range(pEpochs):
        i=0
        for data, label in zip(batch(train_data_x, batchSize),batch(train_data_y, batchSize)):
            #print("Batch: " + str(i+1))
            data = data.as_in_context(myDevice)
            label = label.as_in_context(myDevice)
            data = np.transpose(data,(0,3,1,2))
            with autograd.record():
                output = net(data)
                loss = gluon.loss.SoftmaxCrossEntropyLoss()(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            curr_loss = nd.mean(loss).asscalar()
            i=i+1
        print("Calculating metrics.....")
        #print("Yousef ---> ",train_data_x.shape,train_data_y.shape,batchSize)
        train_accuracy = evaluate_accuracySVHN(train_data_x,train_data_y,batchSize, net)
        test_accuracy = evaluate_accuracySVHN(test_data_x,test_data_y,batchSize, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e+1, curr_loss, train_accuracy, test_accuracy))     
    print("SVHN Training Finished.....")
    memT,cpuT = YN.StartLogger("MXNET","SVHN")

def runModel(dataset,batchSize=128,numClasses=10,epochs=12,learningRate=0.01,momentum=0.5,weightDecay=1e-6):
    if dataset is "mnist":
        RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar10":
        RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar100":
        RunCIFAR100(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "SVHN":
        fname = 'SVHN/%s_32x32.mat'
        RunSVHN(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay,fname) 
    else:
        print("Choose cifar10 or mnist")

def main():
    
    #runModel("mnist",epochs=1)
    #runModel("cifar10",epochs=1)
    runModel("SVHN",epochs=15)
    #runModel("cifar100",epochs=3)
    
    
  
if __name__ == '__main__':
    main()
