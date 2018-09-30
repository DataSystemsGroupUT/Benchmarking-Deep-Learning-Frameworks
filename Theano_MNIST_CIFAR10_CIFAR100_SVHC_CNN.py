#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 14:08:14 2018
@author: nesma
"""
from __future__ import print_function
import os
import time

import numpy as np
import theano
import theano.tensor as T

import pickle
import lasagne
import gzip

import LoggerYN as YN
import scipy.io as sio
import utilsYN as uYN
import warnings
warnings.filterwarnings("ignore", message="Reloaded modules: <module_name>")

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
 
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]    
    
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
    

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def loadDataCIFAR10():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('cifar-10-batches-py/data_batch_'+ str(j+1))
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=X_train,
        Y_train=Y_train.astype('int32'),
        X_test = X_test,
        Y_test = Y_test.astype('int32'),)
    
    
    
    
def loadDataCIFAR100():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('cifar-100-batches-py/train')
      x = d['data']
      y = d['fine_labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('cifar-100-batches-py/test')
    xs.append(d['data'])
    ys.append(d['fine_labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(X_train=X_train,Y_train=Y_train.astype('int32'),X_test = X_test,Y_test = Y_test.astype('int32'),)   
    
def loadDataMNIST():
    from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)


    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(255)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

#    print(X_train.shape)
    global imgRows
    global imgCols
    global imgRGB_Dimensions
    global inputShape
    
    imgRGB_Dimensions = X_train.shape[1]
    imgRows = X_train.shape[2]
    imgCols = X_train.shape[3]
    inputShape = (imgRows, imgCols, imgRGB_Dimensions)
    return X_train, y_train, X_test, y_test


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
    global imgRGB_Dimensions
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
    x_train= theano.shared(x_train).get_value()
    label= theano.shared(y_train).get_value()
    

    x_test = x_test.reshape(len(y_test), imgRows, imgCols, imgRGB_Dimensions)
    x_test= theano.shared(x_test).get_value()
    labelTest= theano.shared(y_test).get_value()

    
    return x_train,label, x_test,labelTest
        
        
        
def modelMNIST(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, imgRGB_Dimensions, imgRows, imgCols),input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.25),num_units=128,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.FlattenLayer(network)
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
    return network





def evalModel(X_test,y_test,batchSize,val_fn):
    #Calculate and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batchSize, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
        
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
    
    
def validateModel(X_val,y_val,batchSize,epoch,train_batches,train_err,start_time,val_fn):
    #pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, batchSize, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
    
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
#    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
#    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))




def RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    X_train, y_train, X_test, y_test = loadDataMNIST()

    # prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # build neural network model
    print("Building model and compiling functions...")
    
    net = modelMNIST(input_var)
    
    # cross-entropy loss for training loss:
    prediction = lasagne.layers.get_output(net)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    
    # SGD
    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.momentum(loss, params, learningRate, momentum)
    
    test_prediction = lasagne.layers.get_output(net, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    
     # training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # test accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

    #validation loss and accuracy
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # loop over all epochs:
    print("MNIST Training Started.....")
    memT,cpuT = YN.StartLogger("Theano","MNIST") 
    for epoch in range(epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, y_train, batchSize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            print('Batch(',train_batches,")",train_err,)
        validateModel(X_train,y_train,batchSize,epoch,train_batches,train_err,val_fn)
    
    evalModel(X_test,y_test,batchSize,val_fn)
    print("MNIST Training Finished.....")
    memT,cpuT = YN.StartLogger("Theano","MNIST")


def modelSVHN(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, imgRGB_Dimensions, imgRows, imgCols),input_var=input_var)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=48, filter_size=(5, 5),pad=2,stride=1,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network,p=0.2)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),pad=2,stride=1,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network,p=0.2)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(5, 5),pad=2,stride=1,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network,p=0.2)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=160, filter_size=(5, 5),pad=2,stride=1,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network,p=0.2)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(5, 5),pad=2,stride=1,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network,p=0.2)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(5, 5),pad=2,stride=1,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network,p=0.2)
    
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(5, 5),pad=2,stride=1,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network,p=0.2)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(5, 5),pad=2,stride=1,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network,p=0.2)
    

    network = lasagne.layers.FlattenLayer(network)    
    network = lasagne.layers.DenseLayer(network,num_units=192,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(network,num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
    return network

def evalModelSVHN(X_test,y_test,batchSize,val_fn):
    #Calculate and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for data, label in zip(batch(X_test, batchSize),batch(y_test, batchSize)):
        data = np.transpose(data,(0,3,1,2))
        err, acc = val_fn(data, label)
        test_err += err
        test_acc += acc
        test_batches += 1
        
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

def validateModelSVHN(X_val,y_val,batchSize,epoch,train_batches,train_err,val_fn):
    val_err = 0
    val_acc = 0
    val_batches = 0
    for data, label in zip(batch(X_val, batchSize),batch(y_val, batchSize)):
        data = np.transpose(data,(0,3,1,2))
        err, acc = val_fn(data, label)
        val_err += err
        val_acc += acc
        val_batches += 1
        
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
#    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
#    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

def RunSVHN(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay,fname):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)

    X_train, y_train, X_test, y_test = loadDataSVHN(fname)

    # prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # build neural network model
    print("Building model and compiling functions...")
    
    net = modelSVHN(input_var)
       
    # cross-entropy loss for training loss:
    prediction = lasagne.layers.get_output(net)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # SGD
    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.momentum(loss, params, learningRate, momentum)
    test_prediction = lasagne.layers.get_output(net, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
        
     # training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # test accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

    #validation loss and accuracy
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # loop over all epochs:
    print("SVHN Training Started.....")
    memT,cpuT = YN.StartLogger("Theano","SVHN") 
    for epoch in range(epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        for data, label in zip(batch(X_train, batchSize),batch(y_train, batchSize)):
            data = np.transpose(data,(0,3,1,2))
            train_err += train_fn(data, label)
            train_batches += 1
            print('Batch(',train_batches,")",train_err,)
        validateModelSVHN(X_train,y_train,batchSize,epoch,train_batches,train_err,val_fn)
        
    evalModelSVHN(X_test,y_test,batchSize,val_fn)
    print("SVHC Training Finished.....")
    memT,cpuT = YN.StartLogger("Theano","SVHN")


def modelCIFAR10(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DropoutLayer(network, p=0.25)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DropoutLayer(network, p=0.25)
    network = lasagne.layers.FlattenLayer(network)

    network = lasagne.layers.DenseLayer(network,num_units=128,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(network,num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
    return network

def RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    # Check if cifar data exists
    if not os.path.exists("./cifar-10-batches-py"):
        print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return

    # Load the dataset
    print("Loading data...")
    data = loadDataCIFAR10()
    
    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    
    # build neural network model
    print("Building model and compiling functions...")
    
    net = modelCIFAR10(input_var)
    
    # cross-entropy loss for training loss:
    prediction = lasagne.layers.get_output(net)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()


    # SGD0
    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.momentum(loss, params, learningRate, momentum)
    
    test_prediction = lasagne.layers.get_output(net, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    
    
     # training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # test accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

    #validation loss and accuracy
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # loop over all epochs:
    print("CIFAR10 Training Started.....")
    memT,cpuT = YN.StartLogger("Theano","CIFAR10") 
    for epoch in range(epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, y_train, batchSize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            print('Batch(',train_batches,")",train_err,)

        validateModel(X_train,y_train,batchSize,epoch,train_batches,train_err,val_fn)
    print("CIFAR10 Training Finished.....")
    memT,cpuT = YN.StartLogger("Theano","CIFAR10")        
    evalModel(X_test,y_test,batchSize,val_fn)
    

def modelCIFAR100(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),stride=1,pad=2,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),stride=1,pad=2,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network, p=0.1)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),stride=1,pad=2,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),stride=1,pad=2,nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network, p=0.25)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=512, filter_size=(3, 3),stride=1,pad=2,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(network, num_filters=512, filter_size=(3, 3),stride=1,pad=2,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2),stride=2,pad=1)
    network = lasagne.layers.DropoutLayer(network, p=0.5)

    
    network = lasagne.layers.FlattenLayer(network)

    network = lasagne.layers.DenseLayer(network,num_units=1024,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(network,num_units=100,nonlinearity=lasagne.nonlinearities.softmax)
    return network


def RunCIFAR100(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    # Check if cifar data exists
    if not os.path.exists("./cifar-100-batches-py"):
        print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return

    # Load the dataset
    print("Loading data...")
    data = loadDataCIFAR100()
    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    

    
    # build neural network model
    print("Building model and compiling functions...")
    
    net = modelCIFAR100(input_var)
    
   
    # cross-entropy loss for training loss:
    prediction = lasagne.layers.get_output(net)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # SGD
    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.momentum(loss, params, learningRate, momentum)
    
    test_prediction = lasagne.layers.get_output(net, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    
    
     # training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # test accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)


    #validation loss and accuracy
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # loop over all epochs:
    print("CIFAR100 Training Started.....")
    memT,cpuT = YN.StartLogger("Theano","CIFAR100")     
    for epoch in range(epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, y_train, batchSize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            print('Batch(',train_batches,")",train_err,)
        validateModel(X_train,y_train,batchSize,epoch,train_batches,train_err,val_fn)
    print("CIFAR100 Training Finished.....")
    memT,cpuT = YN.StartLogger("Theano","CIFAR100")           
    evalModel(X_test,y_test,batchSize,val_fn)


    
    
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
    
    runModel("SVHN",epochs=1)
#    runModel("cifar10",epochs=1)
#    runModel("cifar100",epochs=1)    
#    runModel("mnist",epochs=1)    
    
if __name__ == '__main__':
    main()
