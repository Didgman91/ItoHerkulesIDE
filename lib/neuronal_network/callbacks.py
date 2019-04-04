#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:16:37 2019

@author: itodaiber
"""

import os

import keras

from lib.toolbox import images as ti

class intermediateValidationExport(keras.callbacks.Callback):
    
#    def __init__(self):
#        self.validation_data = None
#        self.model = None
#
#    
#    def set_model(self, model):
#        self.model = model
    
    def on_epoch_end(self, epoch, logs=None):
        print("dir: " + os.getcwd())
        
        path = ["data/neuronal_network/input/validation_data/KW17_thickSP_500_0_0_Intensity_no_pupil_function_layer0000/KW17_thickSP_500_0_0_Intensity_no_pupil_function_layer0330.bmp"]
        image = ti.get_image_as_npy(path)
    
        pred = self.model.predict(image, batch_size=2)
        
        path = self.path_data + self.path_output_validation_data_prediction
        path = "data/neuronal_network/intermediate_data"
        ti.save_4D_npy_as_bmp(pred, ["image_{}.bmp".format(epoch)], path)