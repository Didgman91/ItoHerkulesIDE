#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 14:47:29 2019

@author: itodaiber
"""

import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(1)
#a = np.random.rand(256)
#np.random.seed(654)
#b = np.random.rand(256)


# 1d rect
a = np.zeros(256)
a[150:160] = 1

b = np.zeros(256)
b[160:175] = 1

c = np.convolve(a, b[::-1])

# 2d rect
#a = np.zeros((256, 256))
#a[150:160, 150:160] = 1
#
#b = np.zeros((256, 256))
#b[150:160, 150:160] = 1
#
#c = np.convolve(a.ravel(), b.ravel())

plt.figure()
plt.plot(c)
#plt.imshow(a, cmap='gray')
#plt.axis('off')

plt.show()

size = np.size(c)
posistion_max = np.argmax(c)
shift = (size-1)/2 - posistion_max

print("shift: {}".format(shift))
