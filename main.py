#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:22:54 2018

@author: maxdh
"""

import os

from lib.F2 import F2

from lib.DataIO import mnistLib as mnist

F2.generate_folder_structure()

# -------------------------------------
# IMPORT: Image
# -------------------------------------

#print("# ---------------------------------------------")
#print("# IMPORT: MNIST                                ")
#print("# ---------------------------------------------")
#      
#imagePath = F2.load_MNIST_train_images("data/imagePool/MNIST/samples", range(10))


#print("# ---------------------------------------------")
#print("# IMPORT: NIST                                ")
#print("# ---------------------------------------------")
#
#      
#imageFolder = "../../imagePool/NIST/by_write/hfs_0/f0000_14/c0000_14/"
#
#imageName = os.listdir(imageFolder)
#
#imagePath = []
#
#for name in imageName:
#    extension = os.path.splitext(name)[1]
#    if (extension == ".png"):
#        imagePath = imagePath + [ imageFolder + name]
#      
#imagePath = F2.load_NIST_image(imagePath[:10], True, True, 64, 64)




# -------------------------------------
# F2: Generate and save scatter plate
# -------------------------------------

print("# ---------------------------------------------")
print("# F2: Generate and save scatter plate          ")
print("# ---------------------------------------------")
      
#scatterPlateRandom = F2.create_scatter_plate(F2.get_F2_script_parameter())
scatterPlateRandom = ['data/F2/intermediateData/scatterPlate/ScatterPlateRandomX', 'data/F2/intermediateData/scatterPlate/ScatterPlateRandomY']


# -------------------------------------
# F2: Load scatter plate and calculate specle
# -------------------------------------

print("# ---------------------------------------------")
print("# F2: Load scatter plate and calculate specle  ")
print("# ---------------------------------------------")
      
imageFolder = "data/F2/input/NIST/"

imageName = os.listdir(imageFolder)

imagePath = []

for name in imageName:
    extension = os.path.splitext(name)[1]
    if (extension == ".bmp"):
        imagePath = imagePath + [ imageFolder + name]
        
F2.calculate_propagation(imagePath, scatterPlateRandom)

## -------------------------------------
## IMPORT: npy image
## -------------------------------------
#
#image = np.load('/home/maxdh/Documents/ITO/tmp/Feld.npy')
#
#plt.figure()
#plt.imshow(image, cmap='gray')
#plt.axis('off')
#
#plt.show()
#
## -------------------------------------
## COPY FILES TO SERVER
## -------------------------------------
#
#
## -------------------------------------
## CACLULATE SPECKLE
## -------------------------------------
#
#output, exitCode, t = F2.run_Process("ls", "-lsa")
#
#for i in range(len(output)):
#    print("")
#    print(output[i])
#    
#
#print("time [s]: %.6f" % (t))
#
#
