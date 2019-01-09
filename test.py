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
import statistics

import matplotlib.pyplot as plt

# load images
#imagePath_0_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/variable_fog/0,0/", ["bmp"])
#imagePath_10_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/variable_fog/10,0/", ["bmp"])
#imagePath_100_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/variable_fog/100,0/", ["bmp"])
#imagePath_1000_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/variable_fog/1000,0/", ["bmp"])
#imagePath_10000_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/variable_fog/10000,0/", ["bmp"])
#imagePath_100000_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/variable_fog/100000,0/", ["bmp"])


imagePath_0_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/same_fog/0,0/", ["bmp"])
imagePath_10_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/same_fog/10,0/", ["bmp"])
imagePath_100_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/same_fog/100,0/", ["bmp"])
imagePath_1000_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/same_fog/1000,0/", ["bmp"])
#imagePath_10000_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/same_fog/10000,0/", ["bmp"])
#imagePath_100000_0 = toolbox.get_file_path_with_extension_include_subfolders("./data/memory_effect/input/same_fog/100000,0/", ["bmp"])

    
imagePath_0_0 = [s for s in imagePath_0_0 if "Intensity" in s]
imagePath_10_0 = [s for s in imagePath_10_0 if "Intensity" in s]
imagePath_100_0 = [s for s in imagePath_100_0 if "Intensity" in s]
imagePath_1000_0 = [s for s in imagePath_1000_0 if "Intensity" in s]
#imagePath_10000_0 = [s for s in imagePath_10000_0 if "Intensity" in s]
#imagePath_100000_0 = [s for s in imagePath_100000_0 if "Intensity" in s]



imagePath_0_0.sort()
imagePath_10_0.sort()
imagePath_100_0.sort()
imagePath_1000_0.sort()
#imagePath_10000_0.sort()
#imagePath_100000_0.sort()

image_0_0 = ti.get_image_as_npy(imagePath_0_0)
image_10_0 = ti.get_image_as_npy(imagePath_10_0)
image_100_0 = ti.get_image_as_npy(imagePath_100_0)
image_1000_0 = ti.get_image_as_npy(imagePath_1000_0)
#image_10000_0 = ti.get_image_as_npy(imagePath_10000_0)
#image_100000_0 = ti.get_image_as_npy(imagePath_100000_0)


shift_0 = []
shift_10 = []
shift_100 = []
shift_1000 = []
#shift_10000 = []
#shift_100000 = []
name_10 = []
for i in range(len(image_0_0)):
#    shift_10 += [np.convolve(image_0_0[i,:,:,0].ravel(), image_10_0[i,:,:,0].ravel())]
#    shift_100 += [np.convolve(image_0_0[i,:,:,0].ravel(), image_100_0[i,:,:,0].ravel())]
#    shift_1000 += [np.convolve(image_0_0[i,:,:,0].ravel(), image_1000_0[i,:,:,0].ravel())]
    shift_0 += [signal.fftconvolve(image_0_0[i,:,:,0].ravel(), image_0_0[i,:,:,0].ravel()[::-1])]
    shift_10 += [signal.fftconvolve(image_0_0[i,:,:,0].ravel(), image_10_0[i,:,:,0].ravel()[::-1])]
    shift_100 += [signal.fftconvolve(image_0_0[i,:,:,0].ravel(), image_100_0[i,:,:,0].ravel()[::-1])]
    shift_1000 += [signal.fftconvolve(image_0_0[i,:,:,0].ravel(), image_1000_0[i,:,:,0].ravel()[::-1])]
    
#    shift_10000 += [np.convolve(image_0_0[i,:,:,0].ravel(), image_10000_0[i,:,:,0].ravel())]
#    shift_100000 += [np.convolve(image_0_0[i,:,:,0].ravel(), image_100000_0[i,:,:,0].ravel())]
    name_10 += [toolbox.get_file_name(imagePath_0_0[i])]    

for i in range(int(len(shift_1000))):
    plt.figure()
    
    name = name_10[i][-9:]
    plt.suptitle(name)
    
    plt.subplot(1,5,1)
    plt.plot(shift_0[i])
#    plt.axis('off')
    plt.title("shift 0")
    
    plt.subplot(1,5,2)
    plt.plot(shift_10[i])
#    plt.axis('off')
    plt.title("shift 10")
    
    plt.subplot(1,5,3)
    plt.plot(shift_100[i])
#    plt.axis('off')
    plt.title("shift 100")
#    
    plt.subplot(1,5,4)
    plt.plot(shift_1000[i])
#    plt.axis('off')
    plt.title("shift 10^3")
    
#    plt.subplot(1,5,4)
#    plt.plot(shift_10000[i])
##    plt.axis('off')
#    plt.title("shift 10^4")
#    
#    plt.subplot(1,5,5)
#    plt.plot(shift_100000[i])
##    plt.axis('off')
#    plt.title("shift 10^5")
    
    plt.show()


layer = 0
size = np.size(shift_1000[layer])
posistion_max = np.argmax(shift_100[layer])
shift = (size-1)/2 - posistion_max

print("shift [px]: {}".format(shift))
print("shift [mm]: {}".format(shift*100/256))
    
    
#intensity_1d = shift_10[2][256:,:256].ravel()
#
#f = plt.figure()
#num = len(intensity_1d)
#plt.plot(intensity_1d[-num:])
#plt.show()
#
#f.savefig("foo.pdf", bbox_inches='tight')

#for i in range(len(shift_10[:, 0, 0])):
#    image = ti.convert_3d_npy_to_image(scoreNP[i, :, :],
#                                       invert_color=False)

#def get_correlation(image_path_1, image_path_2):
#    def get_image(image_path):
#        image_path = [s for s in image_path if "Intensity" in s]
#        image_path.sort();
#        image = ti.get_image_as_npy(image_path)
#        
#        return image
#    
#    image_1 = get_image(image_path_1)
#    image_2 = get_image(image_path_2)
#    
#    corr = []
#    for i in range(len(image_2)):
#        corr += [np.corrcoef(image_1[0,:,:,0], image_2[i,:,:,0])]
#    
#    return corr
#
#
#imagePath_origin = imagePath_0_0
#imagePath = [imagePath_10_0,
#             imagePath_100_0]
##             imagePath_1000_0,
##             imagePath_10000_0,
##             imagePath_100000_0]
#
#image = []
#
#for i in range(len(imagePath)):
#    corr = get_correlation(imagePath_origin, imagePath[i])
#    
#    plt.figure()
#    for i in range(len(corr)):
#        plt.subplot(1,len(corr),i+1)
#        plt.imshow(corr[i], cmap='gray')
#        plt.axis('off')
#        plt.title = "{}".format(i+1)
#    
#    plt.show()