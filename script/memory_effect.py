#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:52:52 2019

@author: maxdh
"""

import os

import gc

import numpy as np

from lib.f2 import f2
from lib.toolbox import toolbox
from lib.toolbox import images as ti

from lib.module_base.module_base import module_base

class memory_effect(module_base):
    def __init__(self, **kwds):
       super(memory_effect, self).__init__(name="memory_effect", **kwds)
       
       self.shift_folder_name = "/shfit"

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
        """
        Arguments
        ----
            path: list<string>
                folder paths
                e.g.: ['data/f2/output/speckle/0,0',
                       'data/f2/output/speckle/5,0',
                       'data/f2/output/speckle/10,0']
            
        Folder content (example)
        ----
            path[0]
            - layer0001/Intensity.bmp
            - layer0002/Intensity.bmp
            - layer0003/Intensity.bmp
        """
        def __load(path, search_pattern = "Intensity"):
            """
            Arguments
            ----
                path: string
                    
            Returns
            ----
                all images in the folder at *path* and its subfolders.
            """
            imagePath = toolbox.get_file_path_with_extension_include_subfolders(path, ["bmp"])
            
            # only intesity images
            imagePath = [s for s in imagePath if search_pattern in s]
            
            imagePath.sort()
            
            image = ti.get_image_as_npy(imagePath)
            
            return image
        
        
#        image = []
#        for p in path:
#            image += [__load(p)]
#    #    image += [__load("./data/memory_effect/input/same_fog/0,0/")]
#    #    image += [__load("./data/memory_effect/input/same_fog/10,0/")]
#    #    image += [__load("./data/memory_effect/input/same_fog/100,0/")]
#    #    image += [__load("./data/memory_effect/input/same_fog/1000,0/")]
#        
#        shift = []
#        for i in range(len(image)):
#            shift += [ti.get_cross_correlation_2d(image[i], image[0])]
        
        mm_per_pixel = 50/(4096*2)
        shift_mm = []
        
#        image_0_shift = __load(path[0])
#        
#        size = np.array(np.shape(image_0_shift[0,:,:,0]))
#        position = (size - 1)/2
#        position = position.astype(int)
        
        # iterate shift folders: "5,0", "10,0", ...
        for i in range(1,len(path)):
            folders_layers = os.listdir(path[i])
            folders_layers.sort()
            
            if path[i][-1] == "/":
                    path[i] = path[i][:-1]
            shift_folder = os.path.split(path[i])[-1]
            max_pos = []
            # iterate layers: "layer0001", "layer0002", ...
            for ii in range(len(folders_layers)):
                # image_0
                image_0 = __load(path[0] + "/" + folders_layers[ii])
                size = np.array(np.shape(image_0[0,:,:,0]))
                position = (size - 1)/2
                position = position.astype(int)
                # ~ image_0
                
                image = __load(path[i] + "/" + folders_layers[ii])
                shift = ti.get_cross_correlation_2d(image, image_0)
#                shift = ti.get_cross_correlation_2d(image, np.stack([image_0_shift[ii]]))
                
                
#                if ii % 10 == 0:
#                    correlation_image_file_name = "{}_{}.bmp".format(shift_folder,
#                                                folders_layers[ii])
#                    shift_buffer = shift[0,:,:,0]
#                    shift_buffer_max = np.max(shift_buffer)
#                    ti.save_4D_npy_as_bmp(shift/shift_buffer_max, [correlation_image_file_name],
#                                          self.path_intermediate_data)
                
                max_pos += [ti.get_max_position(shift, relative_position=position)]
                
                gc.collect()
            
            max_pos = np.stack(max_pos)
            max_pos = np.multiply(max_pos, mm_per_pixel)
            shift_mm_buffer = []
            for k in range(len(max_pos)):
                shift_mm_buffer += [max_pos[k,0,0,:]]
            shift_mm += [shift_mm_buffer]
            
            
#       -- old
#        shift = []
#        image_0_shift = __load(path[0])
#        for i in range(len(path)-1):
#            image = __load(path[i+1])
#            shift += [ti.get_cross_correlation_2d(image, image_0_shift)]
            
        
        
#        max_pos = []
#        size = np.array(np.shape(shift[0][0,:,:,0]))
#        position = (size - 1)/2
#        position = position.astype(int)
#        for s in shift:
#            max_pos += [ti.get_max_position(s, relative_position=position)]
#        
#        ~~ old
            
        
        
        
    
        
        
        
        
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
#        folder, path, layer = self.f2_main("", 0)
#        executed_modules += ["f2"]
#        for i in range(1,10):
#            executed_modules += ["f2"]
#            folder, path, layer = self.f2_main("", i*5, False)
        
#        todo: f2 -> class f2
#        self.load_input_from_module()
        
#        self.load_input(f2.pathData + f2.pathOutputSpeckle, "")
        
        path = ["./data/f2/output/speckle/0,0/",
        "./data/f2/output/speckle/5,0/",
        "./data/f2/output/speckle/10,0/",
        "./data/f2/output/speckle/15,0/",
        "./data/f2/output/speckle/20,0/",
        "./data/f2/output/speckle/25,0/",
        "./data/f2/output/speckle/30,0/",
        "./data/f2/output/speckle/35,0/"]
        
        shift = self.evaluate_data(path)
        
#        std = np.std(shift[1])
#        var = np.var(shift[1])
        
        shift = np.stack(shift)
        
        
        shift_path = self.path_ouput + self.shift_folder_name
        
        toolbox.create_folder(shift_path)
        
        for i in range(len(shift)):
            l = shift[i,:,1]
            
            export = []
            for ii in range(len(l)):
                export += [np.array([ii+1, l[ii]])]
            
            export = np.stack(export)
            
            
            np.savetxt(shift_path\
                       + "/shift_{}.csv".format(i*5),
                       export,
                       delimiter = ',',
                       header='fog / m,x-shift',
                       comments='')
        
        executed_modules += [self.module_name]
        
        return executed_modules