#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:22:54 2018

@author: maxdh
"""

from DataIO import mnistLib as mnist

from F2 import F2

# -------------------------------------
# IMPORT: MNIST
# -------------------------------------

images, lables = mnist.load_train_data('/home/maxdh/Documents/ITO/D - Training Data/MNIST/samples/', True)

#mnist.print_data_plt(images[17])

for i in range(5):
    mnist.save_as_bmp(images[i], "output/images/image%04d.bmp" % (i))#(str(i).zfill(3)))

# -------------------------------------
# F2: Generate and save scatter plate
# -------------------------------------



# -------------------------------------
# IMPORT: npy image
# -------------------------------------

image = np.load('/home/maxdh/Documents/ITO/tmp/Feld.npy')

plt.figure()
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.show()

# -------------------------------------
# COPY FILES TO SERVER
# -------------------------------------


# -------------------------------------
# CACLULATE SPECKLE
# -------------------------------------

output, exitCode, t = F2.run_Process("ls", "-lsa")

for i in range(len(output)):
    print("")
    print(output[i])
    

print("time [s]: %.6f" % (t))


