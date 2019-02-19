#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:52:52 2019

@author: maxdh
"""

import os

import gc

import numpy as np

import matplotlib.pyplot as plt

from lib.f2 import f2
from lib.toolbox import toolbox
from lib.toolbox import images as ti

from lib.module_base.module_base import module_base

class memory_effect(module_base):
    def __init__(self, **kwds):
       super(memory_effect, self).__init__(name="memory_effect", **kwds)
       
       self.shift_folder_name = "/shift"

    def f2_main(self, folder, shift, generate_scatter_plate = True):
    #    global executed_modules
    #    executed_modules += ["f2"]
        # -------------------------------------
        # F2
        # -------------------------------------
        toolbox.print_program_section_name("F2")
    
        number_of_layers = 10
        distance = 100  # [mm]
    
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
        def __load(path, search_pattern = ""):
            """
            Arguments
            ----
                path: string
                    
            Returns
            ----
                all images in the folder at *path* and its subfolders.
            """
            imagePath = toolbox.get_file_path_with_extension_include_subfolders(path, ["bmp"])
            
            # e.g. only intesity images
            imagePath = [s for s in imagePath if search_pattern in s]
            
            imagePath.sort()
            
            image = ti.get_image_as_npy(imagePath)
            
            return image
        
        def __save_intersection_image_and_calc_relative_max_pos(img, mm_per_pixel, img_path = "", axis=0):
        
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
            
            if img_path != "":
                f.savefig("{}_intersection.pdf".format(img_path[:-4]))
            
            return relative_pos, std
        
        
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
        
        mm_per_pixel = 25/(4096*2)
        shift_mm = []
        std_mm = []
        
        shift_folder_name = []
        
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
            shift_folder_name += [shift_folder]
            max_pos = []
            std = []
            # iterate layers: "layer0001", "layer0002", ...
            for ii in range(len(folders_layers)):
                # image_0
                image_0 = __load(path[0] + "/" + folders_layers[ii], "Intensity")
                size = np.array(np.shape(image_0[0,:,:,0]))
                position = size/2
                position = position.astype(int)
                # ~ image_0
                
                image = __load(path[i] + "/" + folders_layers[ii], "Intensity")
                shift = ti.get_cross_correlation_2d(image, image_0)
                
                del image_0
                del image
                
                # export cross correlation
                if ii % 1 == 0:
                    correlation_image_file_name = "{}_{}".format(shift_folder,
                                                folders_layers[ii])
                    shift_buffer = shift[0,:,:,0]
                    shift_buffer_max = np.max(shift_buffer)
                    ti.save_4D_npy_as_bmp(shift/shift_buffer_max, [correlation_image_file_name],
                                          self.path_intermediate_data)
                    
                    file_path = self.path_intermediate_data + "/" + correlation_image_file_name + ".bmp"
                    __save_intersection_image_and_calc_relative_max_pos(shift[0,:,:,0], mm_per_pixel, file_path, axis=1)
                # ~ export cross correlation
                
                buffer_pos, buffer_std = ti.get_max_position(shift, relative_position=position)
                
                del shift
                
                max_pos += [buffer_pos]
                std += [buffer_std]
                
                gc.collect()
            
            max_pos = np.stack(max_pos)
            std = np.stack(std)
            
            max_pos = np.multiply(max_pos, mm_per_pixel)
            std = np.multiply(std, mm_per_pixel)
            
            shift_mm_buffer = []
            std_mm_buffer = []
            for k in range(len(max_pos)):
                shift_mm_buffer += [max_pos[k,0,0,:]]
                std_mm_buffer += [std[k,0,0,:]]
            shift_mm += [shift_mm_buffer]
            std_mm += [std_mm_buffer]
        
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
        
        # export to csv
        shift = np.stack(shift_mm)
        std = np.stack(std_mm)
        
        shift_path = self.path_ouput + self.shift_folder_name
        
        toolbox.create_folder(shift_path)
        
        for i in range(len(shift)):
            l = shift[i,:,:]
            s = std[i,:,:]
            
            export = []
            for ii in range(len(l)):
                export += [np.array([ii+1, l[ii,0], l[ii,1], s[ii,0], s[ii,1]])]
            
            export = np.stack(export)
            
            
            np.savetxt(shift_path\
                       + "/shift_{}.csv".format(shift_folder_name[i]),
                       export,
                       delimiter = ',',
                       header='fog / m, x-shift / um, y-shift / um, std_x / um, std_y / um',
                       comments='')
    
        return shift_mm, std_mm, shift_folder_name
    
    def run(self):
        executed_modules = []
        folder_0, path_0, layer_0 = self.f2_main("", 0)
        executed_modules += ["f2"]
        for i in range(1,6):
            executed_modules += ["f2"]
            folder, path, layer = self.f2_main("", 5**i, False)
            
            self.evaluate_data([folder_0, folder])
        
#        todo: f2 -> class f2
#        self.load_input_from_module()
#        self.load_input(f2.pathData + f2.pathOutputSpeckle, "")
        
#        root_folder = "./data/f2/output/speckle"
#        folder = toolbox.get_subfolders("./data/f2/output/speckle")
#        
#        path = []
#        for f in folder:
#            path += [root_folder + "/" + f]
#        
#        path.sort()
##        path = ["./data/memory_effect_new/data/f2/output/speckle/0,0/",
##        "./data/memory_effect_new/data/f2/output/speckle/5,0/"]
#        
#        self.evaluate_data(path)
        
        
        
        executed_modules += [self.module_name]
        
        return executed_modules