#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:58:44 2018

@author: maxdh
"""

import numpy as np

def trainNetwork():
    
    return

def testNetwork():
    
    return

def load_np_images(pathImage):
    "loads the numpy images"
    image = []
    
    for i in pathImage:
        image = image + [np.load(i)]
        
    return image