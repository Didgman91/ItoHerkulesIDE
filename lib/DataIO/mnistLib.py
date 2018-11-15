#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:47:13 2018

@author: maxdh
"""

# How to install: pip install python-mnist

#import random
#import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

from mnist import MNIST

def load_train_data( folder, zipped=False ):
    "load train data of the MNIST database out of the _folder_ folder"
    
    mndata = MNIST(folder)
    mndata.gz = zipped
    
    images, labels = mndata.load_training()
    
    return images, labels
    
def load_testing_data( folder, zipped=False ):
    "load testing data of the MNIST database out of the _folder_ folder"
    
    mndata = MNIST(folder)
    mndata.gz = zipped
    
    images, labels = mndata.load_testing()
    
    return images, labels
    
#def print_data_plt( image, xPixel=0, yPixel=0 ):
#    "print _image_ with matplotlib; _xPixel_ pixels in x; _yPixel_ pixels in y"
#    
#    pixels = get_pixel_array(image, xPixel, yPixel)
#    
#    plt.figure()
#    plt.imshow(pixels, cmap='gray')
#    plt.axis('off')
#    
#    plt.show()
#    return

def save_as_bmp(image, path, xPixel=0, yPixel=0 ):
    pixels = get_pixel_array(image, xPixel, yPixel)
    
    im = Image.fromarray(pixels)
    
    im = im.convert("RGB")
    
    im.save(path)
    
def get_pixel_array(image, xPixel=0, yPixel=0):
    pixels = np.array(image, dtype='uint8')

    if (xPixel == 0 | yPixel == 0):
        buffer = int(np.sqrt(len(pixels)))
        
        if (buffer**2 == len(pixels)):
            xPixel = buffer
            yPixel = buffer
        else:
            print("Error: image isn't squared! Please set _x_ and _y_ pixel number.")
            return
    
    pixels = pixels.reshape((xPixel,yPixel))
    
    return pixels