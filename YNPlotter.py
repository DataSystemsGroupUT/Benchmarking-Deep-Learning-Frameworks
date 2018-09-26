# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:45:01 2018

@author: Youssef
"""

import matplotlib.pyplot as plt


def PlotCPU(fileName,shape="r-"):
    file = open(fileName ,"r") 
    allLines = file.readlines()
    yAxis = []
    for line in allLines:    
        yAxis.append(float(line.split(',')[2].split(":")[1].replace("\n","")))
    plt.plot(yAxis,shape)
    plt.ylabel('CPU Usage %')
    plt.show()

def PlotMemoryNesma(fileName,shape="b-"):
    file = open(fileName ,"r") 
    allLines = file.readlines()
    yAxis = []
    for line in allLines:    
        yAxis.append(float(line.split(',')[1].split(":")[1].replace("\n","")))
    plt.plot(yAxis,shape)
    plt.ylabel('Memory Usage %')
    plt.show()
    
def PlotMemory(fileName,shape="g-"):
    file = open(fileName ,"r") 
    allLines = file.readlines()
    yAxis = []
    for line in allLines:    
        yAxis.append(float(line.split(' ')[1]))
    plt.plot(yAxis,shape)
    plt.ylabel('Memory Usage (MB)')
    plt.show()    


def PlotCPU_Memory(filenameCPU,filenameMemory):
    PlotCPU(filenameCPU)
    PlotMemoryNesma(filenameCPU)
    PlotMemory(filenameMemory)
    
if __name__ == '__main__':
    PlotCPU_Memory("C:/Users/Youssef/Nesma/Benchmarking-Deep-Learning-Frameworks/Result_Cpu/Keras_MNIST_1537948509.txt","C:/Users/Youssef/Nesma/Benchmarking-Deep-Learning-Frameworks/Result_Memory/Keras_MNIST_1537948509.txt")
    PlotCPU_Memory("C:/Users/Youssef/Nesma/Benchmarking-Deep-Learning-Frameworks/Result_Cpu/Keras_CIFAR10_1537949671.txt","C:/Users/Youssef/Nesma/Benchmarking-Deep-Learning-Frameworks/Result_Memory/Keras_CIFAR10_1537949671.txt")