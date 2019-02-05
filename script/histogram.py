#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:33:05 2019

@author: itodaiber
"""

import matplotlib.pyplot as plt

from lib.toolbox import toolbox 
from lib.module_base import module_base
from lib.toolbox import images as ti

class histogram(module_base.module_base):
    def __init__(self, **kwds):
        super(histogram, self).__init__(**kwds)
        self.path_output_histogram = "/histogram"
        
    def creat_histogram(self, path, subfolder=""):
        
        if subfolder == "":
            output_folder = self.path_ouput + self.path_output_histogram
        else:
            output_folder = self.path_ouput + self.path_output_histogram + "/" + subfolder
        
        toolbox.create_folder(output_folder)
        
        hist = ti.get_histogram(path, 256)
        
        # export as csv and as pdf plot
        file_name = []
        for i in range(len(path)):
            file_name += [toolbox.get_file_name(path[i])]
        
        # export as pdf plot
        for i in range(len(hist)):
            toolbox.save_as_csv(hist[i][0], output_folder + "/histogram_{}.csv".format(file_name[i]), file_name[i])
            
            plt.figure()
            plt.plot(hist[i][0])
            
            name = "{}/histogram_{}.pdf".format(output_folder, file_name[i])
            plt.savefig(name)

hist = histogram(name="histogram")

# only for high resolution images!
# otherwise use one of the intrisic load functions:
# - load_input()
# - load_input_from_module()
folder = toolbox.get_subfolders("./data/f2/output/speckle")

for f in folder:
    path = toolbox.get_file_path_with_extension_include_subfolders("./data/f2/output/speckle/{}/".format(f), ["bmp"])
    path = toolbox.get_intersection(path, ["Intensity"], False)
    
    hist.creat_histogram(path, f)