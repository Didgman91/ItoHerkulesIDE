#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 23:41:24 2018

@author: maxdh
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.stats import pearsonr

pred_speckle_9 = np.load('/home/maxdh/Documents/ITO/R - Repositories/ItoDeepSpeckleCorrelation/predictions/pred_speckle_9.npy')
img = Image.open("/home/maxdh/Documents/ITO/R - Repositories/HerkulesIDE/data/F2/input/MNIST/image000004.bmp").convert('L')
img.load()
gt = np.asarray(img, dtype="int32")

pred = pred_speckle_9[0,:,:,0]      # last 0: foreground of the prediction, 1 is the background
predR = Image.fromarray(np.uint8(pred*255)).convert('L')
predR = predR.resize((28,28))
#plt.imshow(predR, cmap='gray')
predR = np.asarray(predR, dtype="int32")


corrcoef = np.corrcoef(gt, predR)

plt.figure()
plt.imshow(corrcoef, cmap='gray')
plt.axis('off')
plt.title("np.corrcoef")
plt.show()


gt1d = []
predR1d = []

for i in range(len(gt[0,:])):
    gt1d = np.concatenate((gt1d, gt[:,i]))
    predR1d = np.concatenate((predR1d, predR[:,i]))

pcc, p = pearsonr(gt1d, predR1d)

print("pearsonr: {}".format(pcc))