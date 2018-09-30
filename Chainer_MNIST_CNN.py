#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:37:12 2018

@author: nesma
"""
from __future__ import print_function
import chainer
import numpy as np
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.links import Convolution2D
from chainer.training import extensions
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
import matplotlib.pyplot as plt
from chainer.datasets import mnist
from chainer import Sequential




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
    
def loadDataMNIST(batchsize):
    if(Dataset == "mnist"):
        train, test = mnist.get_mnist(withlabel=True, ndim=1)
        train_iter = iterators.SerialIterator(train, batchsize)
        test_iter = iterators.SerialIterator(test, batchsize,repeat=False, shuffle=False)
        return train_iter, test_iter
    else:
        pass

class modelMNIST(Chain):
    
    def __init__(self):
        super(modelMNIST, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1,32,ksize=3)
            self.conv2 = L.Convolution2D(32, 64, ksize=3)
            self.fc1 = L.Linear(None,64)
            self.fc2 = L.Linear(64, 10)
      
    def __call__(self, x):
        
        x = x.reshape(-1,1,28,28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  
        x = F.max_pooling_2d(x , ksize= (2, 2))
        x = F.dropout(x, 0.25)
        x = F.flatten(x).reshape(-1,9216)
        x= F.relu(self.fc1(x))
        x = F.dropout(x,0.5)
        x = F.softmax(x)
        return x
    
    




def RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    train_iter, test_iter = loadDataMNIST(batchSize)
    
    model = modelMNIST()

    # Choose an optimizer algorithm
    optimizer = optimizers.MomentumSGD(lr=learningRate, momentum=momentum)
    
    # Give the optimizer a reference to the model so that it
    # can locate the model's parameters.
    optimizer.setup(model)
    
    gpu_id = -1  # Set to -1 for CPU, 0  for GPU
    if gpu_id >= 0:
        model.to_gpu(gpu_id)
        
    batchId = 0  
    while train_iter.epoch < epochs:
    
        batchId += 1
       
        # ---------- One iteration of the training loop ----------
        train_batch = train_iter.next()
        
        image_train, target_train = concat_examples(train_batch, gpu_id)
        
        
        print("Epoch: "+str(train_iter.epoch) , " Batch (",batchId,")")
        # Calculate the prediction of the network
        prediction_train = model(image_train)
    
        # Calculate the loss with softmax_cross_entropy
        loss = F.softmax_cross_entropy(prediction_train, target_train)
    
        # Calculate the gradients in the network
        model.cleargrads()
        loss.backward()
    
        # Update all the trainable parameters
        optimizer.update()
        # --------------------- until here ---------------------
    
        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch
            
            batchId = 0
            # Display the training loss
            print('epoch:{:02d} train_loss:{:.04f} '.format(train_iter.epoch, float(to_cpu(loss.data))), end='')
    
            test_losses = []
            test_accuracies = []
            while True:
                test_batch = test_iter.next()
                image_test, target_test = concat_examples(test_batch, gpu_id)

                # Forward the test data
                prediction_test = model(image_test)
    
                # Calculate the loss
                loss_test = F.softmax_cross_entropy(prediction_test, target_test)
                test_losses.append(to_cpu(loss_test.data))
    
                # Calculate the accuracy
                accuracy = F.accuracy(prediction_test, target_test)
                accuracy.to_cpu()
                test_accuracies.append(accuracy.data)
    
                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break
    
            print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(np.mean(test_losses), np.mean(test_accuracies)))
    


def runModel(dataset,batchSize=128,numClasses=10,epochs=12,learningRate=0.01,momentum=0.5,weightDecay=1e-6):
    if dataset is "mnist":
        RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar10":
        pass
    elif dataset is "cifar100":
        pass
    elif dataset is "SVHN":
        #fname = 'SVHN/%s_32x32.mat'
        pass
    else:
        print("Choose cifar10 or mnist")

def main():
    
    runModel("mnist",epochs=3)
    
    
if __name__ == '__main__':
    main()
