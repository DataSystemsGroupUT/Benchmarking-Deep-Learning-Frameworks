# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:08:11 2018

@author: Youssef
"""
import subprocess
import os
def StartLogger(frameworkName,datasetName):
    print(os.getpid())
    cPid = os.getpid()
    command_memory ="python memLog.py "+ str(cPid) + " " + str(frameworkName) + " " + str(datasetName) + " "
    command_cpu ="python cpuLog.py "+ str(cPid) + " " + str(frameworkName) + " " + str(datasetName) + " "
    memory_task = subprocess.Popen(command_memory, stdout=subprocess.PIPE, shell=True)
    cpu_task = subprocess.Popen(command_cpu, stdout=subprocess.PIPE, shell=True)
    return memory_task,cpu_task

def EndLogger(memory_task,cpu_task):
    memory_task.kill()
    cpu_task.kill()