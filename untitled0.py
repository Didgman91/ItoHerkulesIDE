#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 14:47:29 2019

@author: itodaiber
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#np.random.seed(1)
#a = np.random.rand(256)
#np.random.seed(654)
#b = np.random.rand(256)


def func_1d():
    # 1d rect
    a = np.zeros(256)
    a[150] = 1
    
    b = np.zeros(256)
    b[151] = 1

    c = np.convolve(a, b[::-1])
    
    return c

def func_2d():
    # 2d rect
    a = np.zeros((256, 256))
    a[128, 128] = 1
    
    b = np.zeros((256, 256))
    b[128, 130] = 1
    
    c = signal.fftconvolve(a, b[::-1,::-1])
    
    return c


def plot(c, title):
    dim = len(np.shape(c))

    plt.figure()
    
    if dim == 1:
        plt.plot(c)
    elif dim == 2:
        plt.imshow(c, cmap='gray')
    #plt.axis('off')
    
    plt.title(title)
    
    plt.show()

def get_pos(c, mm_per_pixel):
    size = np.array(np.shape(c))
    posistion_max = np.array(np.unravel_index(np.argmax(c, axis=None), c.shape))
    shift = (size - 1)/2 - posistion_max
    
    dim = len(np.shape(c))
    if dim == 1:
        print("shift: {}".format(shift[0]))
    elif dim == 2:
        print("shift [px]: ({}, {})".format(shift[0], shift[1]))
        print("shift [mm]: ({}, {})".format(shift[0]*mm_per_pixel, shift[1]*mm_per_pixel))
    
    return shift


c = func_2d()

plot(c, "test")
pos = get_pos(c, 1)
