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
import LoggerYN as YN
import scipy.io as sio
import utilsYN as uYN
import numpy as np


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

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
        
def NormalizeData(x_train,x_test):
        x_train /= 255
        x_test /= 255
        return x_train, x_test
    


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
    
    print("MNIST Training Started.....")
    memT,cpuT = YN.StartLogger("Pytorch","MNIST")    
    for epoch in range(1,pEpochs + 1):  
        MNIST_train( MNIST_model, device, MNIST_train_loader, optimizer, epoch)
        MNIST_test(MNIST_model, device, MNIST_test_loader)
    memT,cpuT = YN.StartLogger("Pytorch","MNIST")
    print("MNIST Training Finished.....")
    
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
    #print('\nEpoch: %d' % epoch)
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

    print("CIFAR10 Training Started.....")
    memT,cpuT = YN.StartLogger("Pytorch","CIFAR10")     
    for epoch in range(1, pEpochs+1):
        CIFAR10_train(epoch,criterion,optimizer,CIFAR10_trainloader,device,CIFAR10_model)
        CIFAR10_test(epoch,CIFAR10_testloader,CIFAR10_model,device,criterion)
    memT,cpuT = YN.StartLogger("Pytorch","CIFAR10")
    print("CIFAR10 Training Finished.....")

def CIFAR100_Loader(kwargs):
    
    CIFAR100_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, download=True,
                       transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])), batch_size=pbatchSize, shuffle=True, **kwargs)
    CIFAR100_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])), shuffle=True, **kwargs)
    CIFAR100_classes = (
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
    'bottles', 'bowls', 'cans', 'cups', 'plates',
    'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
    'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple', 'oak', 'palm', 'pine', 'willow',
    'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
    'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')

    return CIFAR100_train_loader, CIFAR100_test_loader,CIFAR100_classes

def CIFAR100_train(epoch,criterion,optimizer,CIFAR100_trainloader,device,CIFAR100_model):
    #print('\nEpoch: %d' % epoch)
    CIFAR100_model.train()
    for batch_idx, (data, targets) in enumerate(CIFAR100_trainloader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        input = CIFAR100_model(data)
        #print(targets.shape)
        loss = F.cross_entropy(input, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(CIFAR100_trainloader.dataset),
                100. * batch_idx / len(CIFAR100_trainloader), loss.item()))

def CIFAR100_test(epoch, CIFAR100_testloader,CIFAR100_model, device,criterion):
    CIFAR100_model.eval()
    test_loss = 0
    correct = 0
    total=0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(CIFAR100_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            input = CIFAR100_model(inputs)
            test_loss = criterion(input, targets)
            test_loss += test_loss.item()
            _, predicted = input.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  
    
    test_loss /= len(CIFAR100_testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(CIFAR100_testloader.dataset),
        100. * correct / len(CIFAR100_testloader.dataset)))


    
class CIFAR100_ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR100_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128,  kernel_size=(3,3),padding=2,stride=1)
        
        self.conv2 = nn.Conv2d( 128, 128, kernel_size=(3, 3),padding=2,stride=1)
        
        self.conv3 = nn.Conv2d( 128, 256, kernel_size=(3, 3),padding=2,stride=1)
        self.conv4 = nn.Conv2d( 256, 256, kernel_size=(3, 3),padding=2,stride=1)
        self.conv5 = nn.Conv2d( 256, 512, kernel_size=(3, 3),padding=2,stride=1)
        self.conv6 = nn.Conv2d( 512, 512, kernel_size=(3, 3),padding=2,stride=1)
         
        self.fc1 = nn.Linear(41472, 192)
        self.fc2 = nn.Linear(192, 100)

            

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.25)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.5)     
        #Flatten  
        x = x.view(x.size(0), -1)
        x= F.relu(self.fc1(x))
        x = F.dropout(x,0.5)   
        x = self.fc2(x)
        y = F.log_softmax(x)
        return y
    
def RunCIFAR100(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    device,kwargs = initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    CIFAR100_trainloader, CIFAR100_testloader, CIFAR100_classes = CIFAR100_Loader(kwargs)
    CIFAR100_model = CIFAR100_ConvNet()
    CIFAR100_model = CIFAR100_model.to(device)
  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(CIFAR100_model.parameters(), lr=pLearningRate, momentum=pMomentum, weight_decay=pWeightDecay)
    
    print("CIFAR100 Training Started.....")
    memT,cpuT = YN.StartLogger("Pytorch","CIFAR100")    
    for epoch in range(1, pEpochs+1):
        CIFAR100_train(epoch,criterion,optimizer,CIFAR100_trainloader,device,CIFAR100_model)
        CIFAR100_test(epoch,CIFAR100_testloader,CIFAR100_model,device,criterion)
    memT,cpuT = YN.StartLogger("Pytorch","CIFAR100")
    print("CIFAR100 Training Finished.....")
    
def loadDataSVHC(fname,extra=False):
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
    x_train = torch.tensor(x_train, dtype=torch.float32)
    label = torch.tensor(y_train,dtype=torch.int64)
    
    x_test = x_test.reshape(len(y_test), imgRows, imgCols, imgRGB_Dimensions)
    x_test = torch.tensor(x_test,  dtype=torch.float32)
    labelTest = torch.tensor(y_test,dtype=torch.int64)
    
    datasetTrainSize = len(y_train)
    datasetTestSize = len(y_test)
    return x_train,label, x_test,labelTest,datasetTrainSize,datasetTestSize


def SVHC_train(epoch,criterion,optimizer,SVHC_trainloader,device,SVHC_model,dsSize):
    print('\nEpoch: %d' % epoch)
    SVHC_model.train()
    batch_idx = 0
    for (data, label) in SVHC_trainloader:
        data, label = data.to(device), label.to(device)
        data = np.transpose(data,(0,3,1,2))
        optimizer.zero_grad()
        input = SVHC_model(data)
        #print(label.long())
        loss = F.cross_entropy(input, label.long())
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), dsSize,
                100. * batch_idx * len(data) / dsSize, loss.item()))
        batch_idx += 1

def SVHC_test(epoch, SVHC_testloader,SVHC_model, device,criterion,dsSize):
    SVHC_model.eval()
    test_loss = 0
    correct = 0
    total=0
    with torch.no_grad():
        for (inputs, targets) in SVHC_testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = np.transpose(inputs,(0,3,1,2))
            input = SVHC_model(inputs)
            test_loss = criterion(input, targets)
            test_loss += test_loss.item()
            _, predicted = input.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  
    
    test_loss /= dsSize
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, dsSize,
        100. * correct / dsSize))
    
    
class SVHC_ConvNet(nn.Module):
    def __init__(self):
        super(SVHC_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48,  kernel_size=(5,5),padding=2,stride=1)
        self.conv2 = nn.Conv2d( 48, 64, kernel_size=(5, 5),padding=2,stride=1)
        self.conv3 = nn.Conv2d( 64, 128, kernel_size=(5, 5),padding=2,stride=1)
        self.conv4 = nn.Conv2d( 128, 160, kernel_size=(5, 5),padding=2,stride=1)
        self.conv5 = nn.Conv2d( 160, 192, kernel_size=(5, 5),padding=2,stride=1)
        self.conv6 = nn.Conv2d( 192, 192, kernel_size=(5, 5),padding=2,stride=1)
        self.conv7 = nn.Conv2d( 192, 192, kernel_size=(5, 5),padding=2,stride=1)
        self.conv8 = nn.Conv2d( 192, 192, kernel_size=(5, 5),padding=2,stride=1)
#        
        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 10)
#        
#        self.conv1 = nn.Conv2d( 3, 32,  kernel_size=(3))
#        self.conv2 = nn.Conv2d( 32, 64, kernel_size=(3, 3))
#        self.fc1 = nn.Linear(2304, 128)
#        self.fc2 = nn.Linear(128, 10)
            

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.2)
        
        x = self.conv2(x)
        bN = nn.BatchNorm2d(num_features=64)
        x = bN(x)
        x = F.relu(x)
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.2)

        x = self.conv3(x)
        bN = nn.BatchNorm2d(num_features=128)
        x = bN(x)
        x = F.relu(x)
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.2)


        x = self.conv4(x)
        bN = nn.BatchNorm2d(num_features=160)
        x = bN(x)
        x = F.relu(x)
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.2)     


        x = self.conv5(x)
        bN = nn.BatchNorm2d(num_features=192)
        x = bN(x)
        x = F.relu(x)
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.2)


        x = self.conv6(x)

        bN = nn.BatchNorm2d(num_features=192)
        x = bN(x)
        x = F.relu(x)
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.2)


        x = self.conv7(x)
        bN = nn.BatchNorm2d(num_features=192)
        x = bN(x)
        x = F.relu(x)
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.2)


        x = self.conv8(x)
        bN = nn.BatchNorm2d(num_features=192)
        x = bN(x)
        x = F.relu(x)
        x = F.max_pool2d(x , kernel_size= (2, 2), stride=2, padding=1)
        x = F.dropout(x,0.2)

        ###########        

        #Flatten  
        x = x.view(x.size(0), -1)
        x= F.relu(self.fc1(x))    
        x = self.fc2(x)
        y = F.log_softmax(x)
        return y
    

def RunSVHC(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay,fname):
    device,kwargs = initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    train_data_x,train_data_y, test_data_x,test_data_y,dsTrainSize,dsTestSize = loadDataSVHC(fname,kwargs)
    
    SVHC_model = SVHC_ConvNet()
    SVHC_model = SVHC_model.to(device)
  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(SVHC_model.parameters(), lr=pLearningRate, momentum=pMomentum, weight_decay=pWeightDecay)
    print("SVHN Training Started.....")
    memT,cpuT = YN.StartLogger("Pytorch","SVHN")   
    for epoch in range(1, pEpochs+1):
        SVHC_train(epoch,criterion,optimizer,zip(batch(train_data_x, batchSize),batch(train_data_y, batchSize)),device,SVHC_model,dsTrainSize)
        SVHC_test(epoch,zip(batch(test_data_x, batchSize),batch(test_data_y, batchSize)),SVHC_model,device,criterion,dsTestSize)
    memT,cpuT = YN.StartLogger("Pytorch","SVHN")
    print("SVHN Training Finished.....")

        
def runModel(dataset,batchSize=128,numClasses=10,epochs=12,learningRate=0.01,momentum=0.5,weightDecay=1e-6):
    if dataset is "mnist":
        RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar10":
        RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar100":
        RunCIFAR100(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)   
    elif dataset is "SVHC":
        fname = 'C:/Users/Youssef/Downloads/Compressed/%s_32x32.mat'
        RunSVHC(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay,fname)    
    else:
        print("Choose cifar10 or mnist")

def main():
    
    runModel("cifar100",epochs=1)
    #runModel("SVHC",epochs=5)
    #runModel("mnist",epochs=1)
    #runModel("cifar10",epochs=1)


if __name__ == '__main__':
    main()

