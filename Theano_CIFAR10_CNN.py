#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 14:40:10 2018

@author: nesma
"""

import sys
import os
import time
import string
import random
import pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

def initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    global Dataset    
    global pbatchSize
    global pnumClassesdef 
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







def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def load_data():
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

        
def model(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.25),num_units=128,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.FlattenLayer(network)
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
    return network

def CIFAR10model(input_var=None):
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
    
    # print the results of the epoch:
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, pEpochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
        
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


def RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    
        # Check if cifar data exists
    if not os.path.exists("./cifar-10-batches-py"):
        print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return

    # Load the dataset
    print("Loading data...")
    data = load_data()
    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    
    
    
    
    # build neural network model
    print("Building model and compiling functions...")
    
    net = CIFAR10model(input_var)
    
   
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
    for epoch in range(epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, y_train, batchSize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            print('bylef: '," - ",train_batches," - ",train_err,)

      #  validateModel(X_val,y_val,batchSize,epoch,train_batches,train_err,start_time,val_fn)
        
    evalModel(X_test,y_test,batchSize,val_fn)



def runModel(dataset,batchSize=128,numClasses=10,epochs=12,learningRate=0.01,momentum=0.5,weightDecay=1e-6):
    if dataset is "mnist":
        pass
        #RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar10":
        RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
   
    else:
        print("Choose cifar10 or mnist")

def main():
    runModel("cifar10",epochs=1,batchSize=500)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
