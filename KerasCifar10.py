#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 14:57:06 2018

@author: nesma
"""

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()