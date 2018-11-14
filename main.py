#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:22:54 2018

@author: maxdh
"""


from lib.F2 import F2

from DataIO import mnistLib as mnist

# -------------------------------------
# IMPORT: MNIST
# -------------------------------------

print("# ---------------------------------------------
print("# IMPORT: MNIST                        ")
print("# ---------------------------------------------

F2.load_MNIST_train_images("data/imagePool/MNIST/samples", range(10))

# -------------------------------------
# F2: Generate and save scatter plate
# -------------------------------------

print("# ---------------------------------------------
print("# F2: Generate and save scatter plate  ")
print("# ---------------------------------------------

F2.generate_folder_structure()
F2.create_scatter_plate(F2.get_F2_script_parameter())


# -------------------------------------
# F2: Load scatter plate and calculate specle
# -------------------------------------

print("# ---------------------------------------------")
print("# F2: Load scatter plate and calculate specle  ")
print("# ---------------------------------------------")


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
