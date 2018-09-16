#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:21:20 2018

@author: nesma
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import psutil
import time
import os
import datetime

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)


class ConvNet(nn.Module):
     def __init__(self):
        super(ConvNet, self).__init__()
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
        x = F.softmax(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('root=root', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
   
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('root=root', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    print("MYDEVICE", device)
    model = ConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    process = psutil.Process(os.getpid())
    print("before memory_percent",process.memory_percent())
    print(psutil.cpu_percent(percpu=True))

    start = time.time()
    print("start",start)
     # args.epochs + 1
    for epoch in range(1,args.epochs + 1):  
        train(args, model, device, train_loader, optimizer, epoch)
        print("IN  memory_percent",process.memory_percent())
        test(args, model, device, test_loader)

    end = time.time()
    print("end", end)
    print("Time Elapsed")
    print(end - start)
    print(str(datetime.timedelta(seconds= end - start)))
    
    
    process = psutil.Process(os.getpid())
    print("after memory_percent",process.memory_percent())
    print(psutil.cpu_percent(percpu=True))  

if __name__ == '__main__':
    main()