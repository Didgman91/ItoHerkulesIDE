#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:19:12 2018

@author: daiberma
"""

import numpy as np

from scipy import signal

from lib.toolbox import toolbox
from lib.toolbox import images as ti

from script import memory_effect as me

import matplotlib.pyplot as plt


# test

shift = me.evaluate_data()
shift = np.stack(shift)
f = plt.figure()

for i in range(len(shift)):
    l = shift[i,:,1]
    plt.plot(l, label="{}".format(i))

plt.title("x-shift")
plt.xlabel("fog / m")
plt.ylabel("calculated shift / mm")
plt.legend(loc="down right")

plt.show()

# ~test

#
#def plot(c, title):
#    dim = len(np.shape(c))
#
#    p = plt.figure()
#    
#    if dim == 1:
#        plt.plot(c)
#    elif dim == 2:
#        plt.imshow(c, cmap='gray')
#    #plt.axis('off')
#    
#    plt.title(title)
#    
#    plt.show()
#    
#    return p
#

