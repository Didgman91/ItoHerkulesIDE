#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:52:52 2019

@author: maxdh
"""

import numpy as np

from lib.f2 import f2
from lib.toolbox import toolbox
from lib.toolbox import images as ti

from lib.module_base.module_base import module_base

class memory_effect(module_base):
    def __init__(self, **kwds):
       super(memory_effect, self).__init__(name="memory_effect", **kwds)
       
       self.shift_folder_name = "shfit/"

    def f2_main(self, folder, shift, generate_scatter_plate = True):
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
    
        #image_path = toolbox.get_file_path_with_extension_include_subfolders(folder,
        #                                                                     ["png"])
    
        #image_path = f2.load_image(image_path[:1], invertColor=True, resize=True, xPixel=64, yPixel=64)
    
        #if image_path == []:
        #    print("F2: no files")
        #    exit()
    
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
    
    #    image_path = toolbox.get_file_path_with_extension(
    #        "data/f2/input/nist/", ["bmp"])
    
        parameters = {"point_source_x_pos": shift,
                      "point_source_y_pos": 0}
    
        f2.calculate_propagation(
            [], scatter_plate_random, number_of_layers, distance, parameters)
    
    #    path = []
    #    layer = []
        folder, path, layer = f2.sortToFolderByLayer(subfolder="{},0/".format(shift))
    
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
    
    def evaluate_data(self, path):
        def __load(path):
            imagePath = toolbox.get_file_path_with_extension_include_subfolders(path, ["bmp"])
            
            # only intesity images
            imagePath = [s for s in imagePath if "Intensity" in s]
            
            imagePath.sort()
            
            image = ti.get_image_as_npy(imagePath)
            
            return image
        
        
        image = []
        for p in path:
            image += [__load(p)]
    #    image += [__load("./data/memory_effect/input/same_fog/0,0/")]
    #    image += [__load("./data/memory_effect/input/same_fog/10,0/")]
    #    image += [__load("./data/memory_effect/input/same_fog/100,0/")]
    #    image += [__load("./data/memory_effect/input/same_fog/1000,0/")]
        
        shift = []
        for i in range(len(image)):
            shift += [ti.get_cross_correlation_2d(image[i], image[0])]
        
        
        max_pos = []
        size = np.array(np.shape(shift[0][0,:,:,0]))
        position = (size - 1)/2
        position = position.astype(int)
        for s in shift:
            max_pos += [ti.get_max_position(s, relative_position=position)]
        
        mm_per_pixel = 100/256
        
        max_pos = np.stack(max_pos)
        max_pos =  np.multiply(max_pos, mm_per_pixel)
    
        shift_mm = []
        for i in range(len(max_pos)):
            shift_mm += [max_pos[i,:,0,:]]
        
        
    #    f = plt.figure()
    #    
    #    for i in range(len(max_pos)):
    #        l = max_pos[i,:,0,1]
    #        plt.plot(l, label="{}".format(i))
    #    
    #    plt.title("x-shift")
    #    plt.xlabel("fog / m")
    #    plt.ylabel("calculated shift / mm")
    #    plt.legend(loc="down right")
    #    
    #    plt.show()
            
        return shift_mm
    
    def run(self):
        executed_modules = []
        folder, path, layer = self.f2_main("", 0)
        executed_modules += ["f2"]
        for i in range(1,10):
            executed_modules += ["f2"]
            folder, path, layer = self.f2_main("", i*5, False)
        
#        todo: f2 -> class f2
#        self.load_input_from_module()
        
        self.load_input(f2.pathData + f2.pathOutputSpeckle)
        
#        path = ["./data/memory_effect/input/same_fog/0,0/",
#        "./data/memory_effect/input/same_fog/5,0/",
#        "./data/memory_effect/input/same_fog/10,0/",
#        "./data/memory_effect/input/same_fog/15,0/",
#        "./data/memory_effect/input/same_fog/20,0/",
#        "./data/memory_effect/input/same_fog/25,0/",
#        "./data/memory_effect/input/same_fog/30,0/",
#        "./data/memory_effect/input/same_fog/35,0/",
#        "./data/memory_effect/input/same_fog/40,0/",
#        "./data/memory_effect/input/same_fog/45,0/"]

        shift = self.evaluate_data(path)
        
#        std = np.std(shift[1])
#        var = np.var(shift[1])
        
        shift = np.stack(shift)
        
        
        
        
        for i in range(len(shift)):
            l = shift[i,:,1]
            
            export = []
            for ii in range(len(l)):
                export += [np.array([ii+1, l[ii]])]
            
            export = np.stack(export)
            
            
            np.savetxt(self.path_ouput\
                       + self.shift_folder_name\
                       + "/shift_{}.csv".format(i*5),
                       export,
                       delimiter = ',',
                       header='fog / m,x-shift',
                       comments='')
        
        executed_modules += [self.module_name]
        
        return executed_modules