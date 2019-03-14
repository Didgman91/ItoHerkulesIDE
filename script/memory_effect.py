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
       
       self.number_of_layers = 500
       self.save_every_no_layer = 10 # saves the first and second one and if layer % save_every_no_layer == 0
       self.distance = 500  # [mm]
       self.mm_per_pixel = 25/(4096)

    def f2_main(self, folder, shift, generate_scatter_plate = True):
    #    global executed_modules
    #    executed_modules += ["f2"]
        # -------------------------------------
        # F2
        # -------------------------------------
        toolbox.print_program_section_name("F2")
    
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
            scatter_plate_random = f2.create_scatter_plate(self.number_of_layers, self.distance)
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
            [], scatter_plate_random, self.number_of_layers, self.distance, parameters=parameters, save_every_no_layer=self.save_every_no_layer)
    
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
    
    def evaluate_data(self, path, save_correlation_image=False):
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
        
        # ---------------------------------------------
        # Memory Effect: evaluate
        # ---------------------------------------------
        toolbox.print_program_section_name(
            "Memory Effect: evaluate")
        print(path)
        
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
        
        def __save_intersection_image_and_calc_relative_max_pos(img, mm_per_pixel, img_path = ""):
        
            size = np.array(np.shape(img))
            middle = (size/2).astype(int)
        
            max_pos, std = ti.get_max_position(img)
            print("max position: {}, {}".format(max_pos[0], max_pos[1]))
            print("std: {}".format(std))
            
            relative_pos = np.subtract(max_pos, middle)
            relative_pos = relative_pos * mm_per_pixel
            
            std = std * mm_per_pixel
            
            
            # axis 0
            intersection = ti.get_intersection(img, axis=0)
            
            maximum = np.max(intersection)
            intersection = intersection / maximum
            
            f = plt.figure()
            plt.plot(intersection)
            plt.ylabel("pixel value / {:.3f}".format(maximum))
            plt.xlabel("pixel number / 1")
            plt.title(img_path)
            plt.text(0,0.900, "x-shift: {:.2f} um".format(relative_pos[0]*1000))
            plt.text(0,0.850, "y-shift: {:.2f} um".format(relative_pos[1]*1000))
            plt.text(0,0.800, "x- std: {:.2f} um".format(std[0]*1000))
            plt.text(0,0.750, "y- std: {:.2f} um".format(std[1]*1000))
            
            if img_path != "":
                f.savefig("{}_intersection_x.pdf".format(img_path[:-4]))
            
            # axis 1
            intersection = ti.get_intersection(img, axis=0)
            
            maximum = np.max(intersection)
            intersection = intersection / maximum
            
            f = plt.figure()
            plt.plot(intersection)
            plt.ylabel("pixel value / {}".format(maximum))
            plt.xlabel("pixel number / 1")
            plt.title(img_path)
            plt.text(0,0.900, "x-shift: {:.2f} um".format(relative_pos[0]*1000))
            plt.text(0,0.850, "y-shift: {:.2f} um".format(relative_pos[1]*1000))
            plt.text(0,0.800, "x- std: {:.2f} um".format(std[0]*1000))
            plt.text(0,0.750, "y- std: {:.2f} um".format(std[1]*1000))
            
            if img_path != "":
                f.savefig("{}_intersection_y.pdf".format(img_path[:-4]))
            
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
        
        shift_mm = []
        std_mm = []
        
        shift_folder_name = []
        
#        image_0_shift = __load(path[0])
#        
#        size = np.array(np.shape(image_0_shift[0,:,:,0]))
#        position = (size - 1)/2
#        position = position.astype(int)
        
        # folder "0,0"
        # subdirs e.d.: layer0001, layer0050, ...
        folder_0_layers = os.listdir(path[0])
        folder_0_layers.sort()
        layers_int = []
        for i in range(len(folder_0_layers)):
            buffer = folder_0_layers[i][len("layer"):]
            layers_int += [int(buffer)]
        
        del folder_0_layers
        
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
                correlation_image_file_name = "{}_{}".format(shift_folder,
                                            folders_layers[ii])
                if save_correlation_image is True:
                    shift_buffer = shift[0,:,:,0]
                    shift_buffer_max = np.max(shift_buffer)
                    ti.save_4D_npy_as_bmp(shift/shift_buffer_max,
                                          [correlation_image_file_name],
                                          self.path_intermediate_data)
                
                file_path = self.path_intermediate_data + "/" + correlation_image_file_name + ".bmp"
                buffer_pos, buffer_std = __save_intersection_image_and_calc_relative_max_pos(shift[0,:,:,0], self.mm_per_pixel, file_path)
                # ~ export cross correlation
                
#                buffer_pos, buffer_std = ti.get_max_position(shift, relative_position=position)
                
                del shift
                
                max_pos += [buffer_pos]
                std += [buffer_std]
                
                gc.collect()
            
            max_pos = np.stack(max_pos)
            std = np.stack(std)
            
            shift_mm += [max_pos]
            std_mm += [std]
            
#            max_pos = np.multiply(max_pos, mm_per_pixel)
#            std = np.multiply(std, mm_per_pixel)
            
#            shift_mm_buffer = []
#            std_mm_buffer = []
#            for k in range(len(max_pos)):
#                shift_mm_buffer += [max_pos[k,0,0,:]]
#                std_mm_buffer += [std[k,0,0,:]]
#            shift_mm += [shift_mm_buffer]
#            std_mm += [std_mm_buffer]
        
        # export
        shift = np.stack(shift_mm)
        std = np.stack(std_mm)
        
        shift_path = self.path_ouput + self.shift_folder_name
        
        toolbox.create_folder(shift_path)

        dist_per_layer = self.distance / self.number_of_layers

        # export to csv
        for i in range(len(shift)):
            l = shift[i,:,:]
            s = std[i,:,:]
            
            export = []
            for ii in range(len(l)):
                export += [np.array([layers_int[ii]*dist_per_layer/1000, l[ii,0], l[ii,1], s[ii,0], s[ii,1]])]
            
            export = np.stack(export)
            
            
            np.savetxt(shift_path\
                       + "/shift_{}.csv".format(shift_folder_name[i]),
                       export,
                       delimiter = ',',
                       header='distance / m, x-shift / um, y-shift / um, std_x / um, std_y / um',
                       comments='')
        
        
        # export pdf plot
        x = np.stack(layers_int) * dist_per_layer
        f = plt.figure()
        file_name_extension = ""
        for i in range(len(shift)):
            file_name_extension += "_" + shift_folder_name[i]
            l = shift[i,:,:] * 1000 # mm -> um
            s = std[i,:,:] * 1000 # mm -> um
            
#            x = arange(1*dist_per_layer, (len(l)+1) * dist_per_layer, dist_per_layer)
            
            plt.plot(x, l[:,1], label=shift_folder_name[i])
#            plt.errorbar(x, l[:,1], s[:,1], label="{} std of max".format(shift_folder_name[i]))
        
        plt.xlabel("distance / mm")
        plt.ylabel("calculated shift / um")
        plt.legend()
        plt.grid(True)
        f.savefig("{}/shift_overview{}.pdf".format(shift_path, file_name_extension))
        
        return shift_mm, std_mm, shift_folder_name
    
    def create_overview(self):
        plot_settings = {'suptitle': 'shift',
                 'xlabel': 'distance / m',
                 'xmul': 1,
                 'ylabel': 'calculated shift / um',
                 'ymul': 1000,  # 2/3 correction of wrong evaluation sam parameter
                 'delimiter': ',',
                 'skip_rows': 1}

        path = "data/memory_effect/output/shift"
        files = toolbox.get_file_path_with_extension(path, ["csv"])
        files.sort()
        
        toolbox.csv_to_plot(files, path + "/shift.pdf", plot_settings=plot_settings,
                    x_column=0, y_column=[2])
    
    def run(self):
        toolbox.copy("script/memory_effect/f2_thick_scatter_plate", "config/f2",
                     replace=True)
        
        executed_modules = []
        folder_0, path_0, layer_0 = self.f2_main("", 0)
        executed_modules += ["f2"]
        for i in range(1,5):
            executed_modules += ["f2"]
            folder, path, layer = self.f2_main("", 5**i, False)
            
            self.evaluate_data([folder_0, folder])
        
        folder, path, layer = self.f2_main("", 50, False)
        self.evaluate_data([folder_0, folder])
        
        folder, path, layer = self.f2_main("", 200, False)
        self.evaluate_data([folder_0, folder])
        
        self.create_overview()
        
        
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