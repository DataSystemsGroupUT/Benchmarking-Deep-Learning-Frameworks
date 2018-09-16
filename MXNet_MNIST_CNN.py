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
mycontext = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

batch_size = 128
num_outputs = 10

#NDArray
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
#        data = data.reshape((-1,)).astype(np.float32)/255
#        return data, label

train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

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
    net.add(gluon.nn.Dense(num_outputs))
    #softmax takes 10 
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mycontext)
#we didn’t have to include the softmax layer 
#because MXNet’s has an efficient function that simultaneously computes 
#the softmax activation and cross-entropy loss

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})




def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(mycontext)
        label = label.as_in_context(mycontext)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]



process = psutil.Process(os.getpid())
print("before memory_percent",process.memory_percent())
print(psutil.cpu_percent(percpu=True))

start = time.time()
print("start",start)


epochs = 12
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        #print("in")
        data = data.as_in_context(mycontext)
        label = label.as_in_context(mycontext)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

end = time.time()

process = psutil.Process(os.getpid())
print("after memory_percent",process.memory_percent())
print(psutil.cpu_percent(percpu=True))
print("end", end)
print("Time Elapsed")
print(str(datetime.timedelta(seconds= end - start)))
print(end - start)

    
    




    






