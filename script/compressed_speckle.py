#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:11:52 2019

@author: itodaiber
"""

import numpy as np

from lib.f2 import f2
from lib.toolbox import toolbox
from lib.toolbox import images as ti
from lib.toolbox import math as tm

from lib.module_base.module_base import module_base

class compressed_speckle(module_base):
    def __init__(self, **kwds):
       super(compressed_speckle, self).__init__(name="compressed_speckle", **kwds)
    
    def get_f2_image_paths(self, extension = ["bmp"]):
        path = f2.pathData + f2.pathInput
        images = toolbox.get_file_path_with_extension(path, extension)
        
        return images
    
    def compress_image(self, image_path):
        """
        image_path: list<string>
        """
        script_filename = "compress.txt"
#        script_path = f2.pathScript + "/" + script_filename
        script_path = "config/f2" + "/" + script_filename
        
        print("script_path: " + script_path)
        
        for path in image_path:
            print(path)
            filename = toolbox.get_file_name(path)
            
            
            output_path = f2.pathData + f2.pathOutput
            
            f_list = [0.5, 5e-2, 5e-3, 5e-4,
                      5e-5, 5e-6, 5e-7]
#            f_list = [5e-5]
            
            for f in f_list:
                f_string = "f_{}".format(f).replace('.', '_')
                folder = output_path + "/" + f_string
                toolbox.create_folder(folder)
                
                fwt_path_pre = folder + "/" + f_string + "_" + filename + "_fwt_pre_c.bmp"
                fwt_path_post = folder + "/" + f_string + "_" + filename + "_fwt_post_c.bmp"
                fwt_path_post_resize = folder + "/" + f_string + "_" + filename + "_fwt_post_c_resize.npy"
                out = folder + "/" + f_string + "_" + filename + ".bmp"
                
                dictionary = {'py_input_image_path': path,
                              'py_output_fwt_pre_c_path': fwt_path_pre,
                              'py_output_fwt_post_c_path': fwt_path_post,
                              'py_output_fwt_post_c_npy_path': fwt_path_post_resize,
                              'py_output_image_path': out,
                              'py_compress_f': f,
                              'py_Pixel_x': 4096,
                              'py_Pixel_y': 4096}
                script = f2.get_f2_script(script_path, dictionary)
                
                s = f2.write_f2_script(script, script_filename)
                
                f2.run_script(s)
                
#                fwt = np.load(fwt_path_post_resize)
#                
#                
#                row_max = np.shape(fwt)[0]-1
#                column_max = np.shape(fwt)[1]-1
#                
#                e_row_max = 0
#                e_column_max = 0
#                for row in range(row_max, 0, -1):
#                    for column in range(column_max, 0, -1):
#                        if fwt[row, column] != 0:
#                            if e_row_max < row:
#                                e_row_max = row
#                            if e_column_max < column:
#                                e_column_max = column
#                                break                
        
    def run(self):
        toolbox.copy("script/compressed_speckle/config/f2/", "config/f2",
                     replace=True)
        
#        input_subfolder = "KW17_thickSP_256mm"
        
        f2.generate_folder_structure()
        
        reload_input = False
        if reload_input == True:
            path = "/home/Grid/itodaiber/KW17/thickSP_256mm_memory_effect_25_mm_fog_d_20_rhon_40_NA_0_12_lam_905_dist_500mm_NAprop_0_01_fog_100m_rhon_0_20/data/f2/output/speckle/0,0/layer0420/Intensity_no_pupil_function_layer0420.bmp"
            toolbox.copy(path, f2.pathData + f2.pathInput)
#            self.load_input(path, input_subfolder)
        
        
        f2.generate_folder_structure()
        files = self.get_f2_image_paths()
        self.compress_image(files)
        
        
        
    
    