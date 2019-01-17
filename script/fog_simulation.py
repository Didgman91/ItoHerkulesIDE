#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:36:31 2019

@author: itodaiber
"""

import numpy as np

from lib.f2 import f2
from lib.toolbox import toolbox
from lib.toolbox import images as ti

#from lib.module_base.module_base import module_base
#
#class fog_simulation(module_base):
#    def __init__(self, **kwds):
#        super(fog_simulation, self).__init__(**kwds)
    
def f2_main(folder, generate_scatter_plate = True):
#    global executed_modules
#    executed_modules += ["f2"]
    # -------------------------------------
    # F2
    # -------------------------------------
    toolbox.print_program_section_name("F2")

    number_of_layers = 100
    distance = 100000  # [mm]

    f2.generate_folder_structure()

    # -------------------------------------
    # IMPORT: Image
    # -------------------------------------
    #toolbox.print_program_section_name("IMPORT: MNIST images")
    #
    #image_path = F2.load_mNIST_train_images("data/imagePool/MNIST/samples", range(10))

    toolbox.print_program_section_name("IMPORT: NIST images")

    image_path = toolbox.get_file_path_with_extension_include_subfolders(folder,
                                                                         ["png"])

    image_path = f2.load_image(image_path[:1], invertColor=True, resize=True, xPixel=64, yPixel=64)

    if image_path == []:
        print("F2: no files")
        exit()

    # ---------------------------------------------
    # F2: Generate and save scatter plate
    # ---------------------------------------------
    toolbox.print_program_section_name("F2: Generate and save scatter plate")

    if generate_scatter_plate is True:
        scatter_plate_random = f2.create_scatter_plate(number_of_layers, distance)
    else:
        scatter_plate_random = ['data/f2/intermediate_data/scatter_plate/scatter_plate_random_x', 'data/f2/intermediate_data/scatter_plate/scatter_plate_random_y']

    # -------------------------------------------------
    # F2: Load scatter plate and calculate speckle
    # -------------------------------------------------
    toolbox.print_program_section_name(
        "F2: Load scatter plate and calculate speckle")

    image_path = toolbox.get_file_path_with_extension(
        "data/f2/input/nist/", ["bmp"])

    f2.calculate_propagation(
        image_path, scatter_plate_random, number_of_layers, distance)

#    path = []
#    layer = []
    folder, path, layer = f2.sortToFolderByLayer()

    #import matplotlib.pyplot as plt
    #
    # for i in range(len(images)):
    #    plt.figure()
    #    plt.imshow(images[i], cmap='gray')
    #    plt.axis('off')
    #    plt.title(image_path[i])
    #
    #
    #    plt.show()
    

    return folder, path, layer

def run():
    image_pool_path = "../imagePool/NIST/by_write/hsf_2/f1000_45/"
    folder, path, layer = f2_main(image_pool_path)