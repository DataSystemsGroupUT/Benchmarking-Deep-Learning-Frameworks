#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  22  19:21:20 2018
@author: nesma
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import datasets
import psutil
import time
import os
import datetime



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
     
   
    device = "cpu" 
    
    kwargs = {}
    
    return device,kwargs


def MNIST_Loader(kwargs):
    MNIST_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('C:/Users/Youssef/Nesma/Benchmarking-Deep-Learning-Frameworks/data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=pbatchSize, shuffle=True, **kwargs)
    MNIST_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('C:/Users/Youssef/Nesma/Benchmarking-Deep-Learning-Frameworks/data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), shuffle=True, **kwargs)
    
    
    return MNIST_train_loader, MNIST_test_loader #  MNIST_train_loader -> x_train,y_train  # MNIST_test_loader -> x_test,y_test






def MNIST_train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        input = model(data)
        loss = F.cross_entropy(input, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def MNIST_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            input = model(data)
            test_loss += F.cross_entropy(input, target, reduction='sum').item() # sum up batch loss
            pred = input.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

class MNIST_ConvNet(nn.Module):
     def __init__(self):
        super(MNIST_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d( 1, 32,  kernel_size=(3))
        self.conv2 = nn.Conv2d( 32, 64, kernel_size=(3, 3))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(9216, 64)
        self.fc2 = nn.Linear(64, 10)
            

     def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  
        x = F.max_pool2d(x , kernel_size= (2, 2))
        x = F.dropout(x, 0.25)
        #Flatten  
        x = x.view(x.size(0), -1)
        x= F.relu(self.fc1(x))        
        x = F.dropout(x,0.5)
        x = self.fc2(x)
        y = F.log_softmax(x)
        #log_softmax it’s faster and has better numerical properties).
        return y


    
def RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    device, kwargs = initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    MNIST_train_loader, MNIST_test_loader = MNIST_Loader(kwargs)
    MNIST_model = MNIST_ConvNet().to(device)
    optimizer = optim.SGD(MNIST_model.parameters(), lr=pLearningRate, momentum=pMomentum, weight_decay=pWeightDecay)

    StartMonitoring()
    
    for epoch in range(1,pEpochs + 1):  
        MNIST_train( MNIST_model, device, MNIST_train_loader, optimizer, epoch)
        MNIST_test(MNIST_model, device, MNIST_test_loader)

    
    EndMonitoring()    
    

def CIFAR10_Loader(kwargs):
    
    CIFAR10_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('C:/Users/Youssef/Nesma/Benchmarking-Deep-Learning-Frameworks/data', train=True, download=True,
                       transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])), batch_size=pbatchSize, shuffle=True, **kwargs)
    CIFAR10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('C:/Users/Youssef/Nesma/Benchmarking-Deep-Learning-Frameworks/data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])), shuffle=True, **kwargs)
    CIFAR10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return CIFAR10_train_loader, CIFAR10_test_loader,CIFAR10_classes

def CIFAR10_train(epoch,criterion,optimizer,CIFAR10_trainloader,device,CIFAR10_model):
    print('\nEpoch: %d' % epoch)
    CIFAR10_model.train()
    for batch_idx, (data, targets) in enumerate(CIFAR10_trainloader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        input = CIFAR10_model(data)
        loss = F.cross_entropy(input, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(CIFAR10_trainloader.dataset),
                100. * batch_idx / len(CIFAR10_trainloader), loss.item()))
    
def CIFAR10_test(epoch, CIFAR10_testloader,CIFAR10_model, device,criterion):
    CIFAR10_model.eval()
    test_loss = 0
    correct = 0
    total=0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(CIFAR10_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            input = CIFAR10_model(inputs)
            test_loss = criterion(input, targets)
            test_loss += test_loss.item()
            _, predicted = input.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  
    
    test_loss /= len(CIFAR10_testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(CIFAR10_testloader.dataset),
        100. * correct / len(CIFAR10_testloader.dataset)))
      


class CIFAR10_ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d( 3, 32,  kernel_size=(3))
        self.conv2 = nn.Conv2d( 32, 64, kernel_size=(3, 3))
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)
            

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  
        x = F.max_pool2d(x , kernel_size= (2, 2))
        x = F.dropout(x, 0.25)
        #Flatten  
        x = x.view(x.size(0), -1)
        x= F.relu(self.fc1(x))    
        x = F.dropout(x,0.5)
        x = self.fc2(x)
        y = F.log_softmax(x)
        return y
    
def RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    device,kwargs = initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    CIFAR10_trainloader, CIFAR10_testloader, CIFAR10_classes = CIFAR10_Loader(kwargs)
    CIFAR10_model = CIFAR10_ConvNet()
    CIFAR10_model = CIFAR10_model.to(device)
  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(CIFAR10_model.parameters(), lr=pLearningRate, momentum=pMomentum, weight_decay=pWeightDecay)
    StartMonitoring()
    
    for epoch in range(1, pEpochs+1):
        CIFAR10_train(epoch,criterion,optimizer,CIFAR10_trainloader,device,CIFAR10_model)
        CIFAR10_test(epoch,CIFAR10_testloader,CIFAR10_model,device,criterion)
    
    EndMonitoring()



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
        

        
def runModel(dataset,batchSize=128,numClasses=10,epochs=12,learningRate=0.01,momentum=0.5,weightDecay=1e-6):
    if dataset is "mnist":
        RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar10":
        RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)     
    else:
        print("Choose cifar10 or mnist")

def main():
    #runModel("mnist",epochs=1)
    runModel("cifar10",epochs=1)


if __name__ == '__main__':
    main()
