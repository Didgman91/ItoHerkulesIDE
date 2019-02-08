#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:19:12 2018

@author: daiberma
"""

import os

import numpy as np

from scipy import signal

from lib.toolbox import toolbox
from lib.toolbox import images as ti

from script import memory_effect as me

import matplotlib.pyplot as plt


from lib.latex import documentation


from lib.module_base import module_base


#
#image_path = toolbox.get_file_path_with_extension("/home/maxdh/Documents/tmp", ["bmp"])
#image_path.sort()
#
#import imageio
#
#for ip in image_path:
#    image = imageio.imread(ip, 'bmp')
#
#    histogram, bin_edges = np.histogram(image[:,:,1], 256)
##    
#    file_name = ip[-13:-4]
##    
##    f = plt.figure()
##    plt.plot(histogram)
##    plt.title(ip[-13:])
##    name = "{}.pdf".format(file_name)
##    f.savefig(name, bbox_inches="tight")
#    
#    export = []
#    hist_max = histogram.max()
#    for ii in range(len(histogram)):
#        export += [np.array([ii+1, histogram[ii] / hist_max])]
#    
#    np.savetxt("./hist/hist_{}.csv".format(file_name),
#               export,
#               delimiter = ',',
#               header='8-bit intesity,normalized histogram',
#               comments='')
#




import cv2

mm_per_pixel = 25.0/(4096*2)

def get_intersection(img, mm_per_pixel, axis = 0):
    size = len(img[0])
    middle = int((size)/2)-1
    
    max_pos, std = ti.get_max_position(img)
    print("max position: {}, {}".format(max_pos[0], max_pos[1]))
    print("std: {}".format(std))
    
    relative_pos = max_pos - np.array([middle, middle])
    relative_pos = relative_pos * mm_per_pixel
    print("relative max position [um]: {}, {}".format(relative_pos[0]*1000, relative_pos[1]*1000))
    
    if axis == 0:
        return img[:,middle], relative_pos, std
    else:
        return img[middle,:], relative_pos, std
    
    
    
    

def show_intersection_and_calc_relative_max_pos(img, mm_per_pixel, img_path = "", axis=0, save_img=False):
    #cv2.imshow('image', img)
    
#    plt.imshow(img, cmap = 'plasma', interpolation = 'bicubic')
#    plt.show()
    
    size = np.array(np.shape(img))
    middle = (size/2).astype(int)

    max_pos, std = ti.get_max_position(img)
    print("max position: {}, {}".format(max_pos[0], max_pos[1]))
    print("std: {}".format(std))
    
    relative_pos = np.subtract(max_pos, middle)
    relative_pos = relative_pos * mm_per_pixel
    
    intersection = ti.get_intersection(img, axis=axis)
    
    maximum = np.max(intersection)
    intersection = intersection / maximum
    
    f = plt.figure()
    plt.plot(intersection)
    plt.ylabel("pixel value / {}".format(maximum))
    plt.xlabel("pixel number / 1")
    plt.title(img_path)
    plt.text(0,0.900, "x-shift: {:.2f} um".format(relative_pos[0]*1000))
    plt.text(0,0.850, "y-shift: {:.2f} um".format(relative_pos[1]*1000))
    plt.text(0,0.800, "x- std: {:.2f} um".format(std[0]*mm_per_pixel*1000))
    plt.text(0,0.750, "y- std: {:.2f} um".format(std[1]*mm_per_pixel*1000))
    plt.show()
    
    if save_img is True:
        f.savefig("{}_intersection.pdf".format(img_path[:-4]))
    
#    min_max_x = [int(max_pos[0]-size/10), int(max_pos[0]+size/10)]
#    min_max_y = [int(max_pos[1]-size/10), int(max_pos[1]+size/10)]
#    
#    plt.imshow(img[min_max_x[0]:min_max_x[1], min_max_y[0]:min_max_y[1]], cmap = 'plasma', interpolation = 'bicubic')
#    plt.show()

#path = "/tmp/memory_effect/5,0_layer0091.bmp.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#path = "/tmp/memory_effect/10,0_layer0001.bmp.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#path = "/tmp/memory_effect/10,0_layer0011.bmp.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#path = "/tmp/memory_effect/10,0_layer0021.bmp.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#path = "/tmp/memory_effect/10,0_layer0031.bmp.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#path = "/tmp/memory_effect/10,0_layer0041.bmp.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#path = "/tmp/memory_effect/0,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#path = "/tmp/memory_effect/5,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#path = "/tmp/memory_effect/10,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#path = "/tmp/memory_effect/45,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#    
#    
## me_new
#path = "/tmp/memory_effect_new/data/f2/output/speckle/0,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
##print()
##path = "/tmp/memory_effect_new/data/f2/output/speckle/5,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
##show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#print()
#path = "/tmp/memory_effect_new/data/f2/output/speckle/10,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#print()
#path = "/tmp/memory_effect_new/data/f2/output/speckle/20,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#print()
#path = "/tmp/memory_effect_new/data/f2/output/speckle/30,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#print()
#path = "/tmp/memory_effect_new/data/f2/output/speckle/40,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)
#
#print()
#path = "/tmp/memory_effect_new/data/f2/output/speckle/50,0/layer0001/Intensity_no_pupil_function_layer0001.bmp"
#show_intersection_and_calc_relative_max_pos(path, mm_per_pixel)


#path = "/home/itodaiber/Documents/ITO/R - Repositories/HerkulesIDE/data/memory_effect/intermediate_data"
#path = "/home/itodaiber/Documents/ITO/3 - Datasets/KW10/memory_effect_25mm/"

#path = "/home/itodaiber/Documents/ITO/3 - Datasets/KW11/memory_effect_25_mm_fog_5_m/f2/25,0"
#path = "/home/itodaiber/Documents/ITO/3 - Datasets/KW11/memory_effect_25_mm_fog_5_m/memory_effect"

#path = "/home/itodaiber/Documents/ITO/3 - Datasets/KW11/memory_effect_25_mm_fog_20_m/f2/125,0/"
#path = "/home/itodaiber/Documents/ITO/3 - Datasets/KW11/memory_effect_25_mm_fog_20_m/memory_effect/"
        
#path = "/home/itodaiber/Documents/ITO/3 - Datasets/KW11/memory_effect_25_mm_fog_100_m/f2/25,0/"
#path = "/home/itodaiber/Documents/ITO/3 - Datasets/KW11/memory_effect_25_mm_fog_100_m/memory_effect/"

path = "/home/itodaiber/Documents/ITO/3 - Datasets/memory_effect_25_mm_thin_scatter_medium_NA_0.05/memory_effect/"

file = toolbox.get_file_path_with_extension(path, ["bmp"])
#file = toolbox.get_intersection(file, ["Intensity"], False)
file.sort()

for f in file:    
    print("---- new image ----")
    img = cv2.imread(f,0)
    print(f)
    show_intersection_and_calc_relative_max_pos(img, mm_per_pixel, f, axis=1, save_img=True)



#image = imageio.imread(path, 'bmp')

#f = plt.figure()
#plt.plot(image[:,4096,0])
#plt.show()

#toolbox.save_as_csv(image[:,4096,0], "./tmp/intersection_y_4096.csv", ["x","y"])


#all_bmp = toolbox.get_file_path_with_extension_include_subfolders("/home/itodaiber/Documents/ITO/imagePool/NIST/by_write/hsf_2/f1000_45/", ["png"])
#ready = toolbox.read_file_lines("ready.txt")
#ready = list(set(ready))
#diff = toolbox.get_relative_complement(all_bmp, ready, False)


## ------------------------------mod tex files (add plots) ------------------------------
#path_plot = "lib/latex/document/tex/plot/plot.tex"
#toolbox.copy("lib/latex/document/tex/ORG_plot.tex", path_plot)
#
#d = {'py_file_name':'shift_5.csv',
#     'py_addlegend': 'shift 5\\,mm'}
#path_addplot_1 = "lib/latex/document/tex/plot/plot_addplot_{}.tex".format(1)
#toolbox.copy("lib/latex/document/tex/ORG_plot_addplot.tex", path_addplot_1)
#toolbox.replace_in_file(path_addplot_1, d)
#
#
#d = {'py_addplot':"\\input{" + path_addplot_1.replace("lib/latex/document/", "") + "}" + "\npy_addplot"}
#toolbox.replace_in_file(path_plot, d)
#
#d = {'py_file_name':'shift_10.csv',
#     'py_addlegend': 'shift 10\\,mm'}
#path_addplot_2 = "lib/latex/document/tex/plot/plot_addplot_{}.tex".format(2)
#toolbox.copy("lib/latex/document/tex/ORG_plot_addplot.tex", path_addplot_2)
#toolbox.replace_in_file(path_addplot_2, d)
#
#d = {'py_addplot':"\\input{" + path_addplot_2.replace("lib/latex/document/", "") + "}"}
#toolbox.replace_in_file(path_plot, d)
#
#documentation.generate_documentation()
## mod tex -------------------------------------------------------------------------------



## test: Memory effect -------------------------------------------------------------------
#
#path = ["./data/memory_effect/input/same_fog/0,0/",
#        "./data/memory_effect/input/same_fog/5,0/",
#        "./data/memory_effect/input/same_fog/10,0/",
#        "./data/memory_effect/input/same_fog/15,0/",
#        "./data/memory_effect/input/same_fog/20,0/",
#        "./data/memory_effect/input/same_fog/25,0/",
#        "./data/memory_effect/input/same_fog/30,0/",
#        "./data/memory_effect/input/same_fog/35,0/",
#        "./data/memory_effect/input/same_fog/40,0/",
#        "./data/memory_effect/input/same_fog/45,0/"]
#
#shift = me.evaluate_data(path)
#shift = np.stack(shift)
#
#
#for i in range(len(shift)):
#    l = shift[i,:,1]
#    
#    export = []
#    for ii in range(len(l)):
#        export += [np.array([ii+1, l[ii]])]
#    
#    export = np.stack(export)
#    
#    np.savetxt("data/memory_effect/shift_{}.csv".format(i*5),
#               export,
#               delimiter = ',',
#               header='fog / m,x-shift',
#               comments='')
#    
#
#


#f = plt.figure()
#
#for i in range(len(shift)):
#    l = shift[i,:,1]
#    plt.plot(l, label="{} mm".format(i*5))
#
#plt.title("x-shift")
#plt.xlabel("fog / m")
#plt.ylabel("calculated shift / mm")
#plt.legend(loc="down right")
#
#plt.show()
#f.savefig("x-shift.pdf", bbox_inches="tight")
#
#
#f = plt.figure()
#for i in range(len(shift)):
#    l = shift[i,:,0]
#    plt.plot(l, label="{} mm".format(i*5))
#
#plt.title("y-shift")
#plt.xlabel("fog / m")
#plt.ylabel("calculated shift / mm")
#plt.legend(loc="down right")
#
#plt.show()
#f.savefig("y-shift.pdf", bbox_inches="tight")


## ~test-------------------------------------------------------------------------------------

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

