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

max_epoch = 3
batchsize = 1024

    
    
train, test = mnist.get_mnist(withlabel=True, ndim=1)

train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize,
                                     repeat=False, shuffle=False)

class MyNetwork(Chain):
    def __init__(self):
        super(MyNetwork, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1,32,ksize=3)
            self.conv2 = L.Convolution2D(32, 64, ksize=3)
            self.fc1 = L.Linear(None,64)
            self.fc2 = L.Linear(64, 10)
      
    def __call__(self, x):
        
        #print(x.shape)
        x = x.reshape(-1,1,28,28)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))  
        #print(x.shape)
        x = F.max_pooling_2d(x , ksize= (2, 2))
        #print(x.shape)
        x = F.dropout(x, 0.25)
        #print(x.shape)
        print(x.shape)
        #x = F.flatten(x).reshape(batchsize,-1)
        x = F.flatten(x).reshape(-1,9216)
        print(x.shape)

        print("***",x.shape)
        x= F.relu(self.fc1(x))
        #print("====",x.shape)        
        x = F.dropout(x,0.5)
        x = F.softmax(x)
        return x
    
    
model = MyNetwork()

# Choose an optimizer algorithm
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)

# Give the optimizer a reference to the model so that it
# can locate the model's parameters.
optimizer.setup(model)

gpu_id = -1  # Set to -1 for CPU, 0  for GPU
if gpu_id >= 0:
    model.to_gpu(gpu_id)
    
    
    
while train_iter.epoch < max_epoch:

    # ---------- One iteration of the training loop ----------
    train_batch = train_iter.next()
    
    image_train, target_train = concat_examples(train_batch, gpu_id)
    
    
#    while(image_train.shape[0] != batchsize):
#        train_batch = train_iter.next()
#        image_train, target_train = concat_examples(train_batch, gpu_id)
#    
    print("----" + str(image_train.shape) + "--" + str(train_iter.epoch))
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

        # Display the training loss
        print('epoch:{:02d} train_loss:{:.04f} '.format(
            train_iter.epoch, float(to_cpu(loss.data))), end='')

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

        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            np.mean(test_losses), np.mean(test_accuracies)))


serializers.save_npz('my_mnist.model', model)   

# Create an instance of the network you trained
model = MyNetwork()

# Load the saved parameters into the instance
serializers.load_npz('my_mnist.model', model)

# Get a test image and label
x, t = test[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.savefig('7.png')
print('label:', t)


# Change the shape of the minibatch.
# In this example, the size of minibatch is 1.
# Inference using any mini-batch size can be performed.

print(x.shape, end=' -> ')
x = x[None, ...]
print(x.shape)

# Forward calculation of the model by sending X
y = model(x)

# The result is given as Variable, then we can take a look at the contents by the attribute, .data.
y = y.data

# Look up the most probable digit number using argmax
pred_label = y.argmax(axis=1)

print('predicted label:', pred_label[0])
