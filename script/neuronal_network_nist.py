#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:23:48 2019

@author: itodaiber
"""


from lib.toolbox import toolbox
from lib.toolbox import images as ti

from lib.neuronal_network import neuronal_network

from config.neuronal_network.model.model import model_deep_specle_correlation_class

from lib.module_base.module_base import module_base

from keras import backend as K

class data_struct():
    def __init__(self, training_data_path, ground_truth_data_path):
        self.__training_data_path = training_data_path
        self.__ground_truth_data_path

class memory_effect_nn(module_base):
    def __init__(self, **kwds):
       super(memory_effect_nn, self).__init__(name="memory_effect_nn", **kwds)
    
    def nn_init(self):
#        K.set_floatx('float32')
        neuronal_network.set_tf_gpu_fraction(0.12)
#       neuronal_network.set_tf_gpu_allow_growth()


    def nn_main(self, external_module_paths, external_module_label):
        self.nn_init()
        def get_image_paths_from_other_runs(external_folders, external_label, filter_str="Intensity"):
            """
            Arguments
            ----
                external_folders: list<string>
                    list of paths of the external folders
            """
            
            def __filter(l, search_pattern):
                filterd_list = [s for s in l if search_pattern in s]
                
                return filterd_list
            
            
            path = []
            relative_path = []
            for i in range(len(external_folders)): # f2/output/speckle
                path_buffer = toolbox.get_file_path_with_extension_include_subfolders(external_folders[i], ["bmp"])
                if filter_str != "":
                    path_buffer = __filter(path_buffer, filter_str)
                path += path_buffer
                for b in path_buffer:
                    filename = toolbox.get_file_name(b, with_extension=True)
                    relative_path += [ external_label[i] + b[len(external_folders[i])+1:-len(filename)-1-len("_layer0000")]]
                
            
            return path, relative_path
        
        
                
        
        # ---------------------------------------------
        # NEURONAL NETWORK: load data
        # ---------------------------------------------
        toolbox.print_program_section_name("NEURONAL NETWORK: load data")
    
        print("the model is loading...")
        m = model_deep_specle_correlation_class()
    
        pixel = 64
        resize = False
        nn = neuronal_network.neuronal_network_class(m.get_model(pixel))
#        nn = neuronal_network.neuronal_network_class([])
        print("done")
        
        # nn.load data from file system
        reload_images = True
        if reload_images is True:
            print("the training and test datasets are loading...")
            
            ground_truth_path = "/data/F2/input/NIST"
            path_ground_truth = []
            for p in external_module_paths:
                path_ground_truth = [p + ground_truth_path]
            
            image_gt_path, relative_gt_path = get_image_paths_from_other_runs(path_ground_truth,
                                                                        external_module_label,
                                                                        filter_str="")
            
            for i in range(len(relative_gt_path)):
                relative_gt_path[i] = relative_gt_path[i].replace('/','_')
                relative_gt_path[i] = relative_gt_path[i].replace(',', '_')
            
            
            
            image_path = "/data/F2/output/speckle"
            buffer = []
            for p in external_module_paths:
                buffer = [p + image_path]
            image_path, relative_path = get_image_paths_from_other_runs(buffer,
                                                                        external_module_label)
            
            
            
            for i in range(len(relative_path)):
                relative_path[i] = relative_path[i].replace('/','_')
                relative_path[i] = relative_path[i].replace(',', '_')
                    
            
            gt_1, gt_2, r_element = toolbox.split_list_randome(image_gt_path,
                                                             percentage=90)
            
            
            gt_1_filename = toolbox.get_file_name(gt_1)
            gt_2_filename = toolbox.get_file_name(gt_2)
            train_image = toolbox.get_relative_complement(image_path,
                                                          gt_2_filename,
                                                          str_diff_exact=False)
            validation_image = toolbox.get_intersection(image_path,
                                                        gt_2_filename,
                                                        str_diff_exact=False)
            
            nn.load_ground_truth_data(gt_1,
                                      resize=resize,
                                      x_pixel=pixel, y_pixel=pixel)
            nn.load_validation_ground_truth_data(gt_2,
                                      resize=resize,
                                      x_pixel=pixel, y_pixel=pixel)
            
            for gt in gt_1_filename:
                buffer = toolbox.get_intersection(train_image, [gt],
                                                  str_diff_exact=False)
                nn.load_training_data(buffer, gt,
                                      resize=resize,
                                      x_pixel=pixel, y_pixel=pixel)
            for gt in gt_2_filename:
                buffer = toolbox.get_intersection(validation_image, [gt],
                                                  str_diff_exact=False)
                nn.load_validation_data(buffer, gt,
                                        resize=resize,
                                        x_pixel=pixel, y_pixel=pixel)
            
            
            
#            # split data sets: parameter: relative_path change -> shift change e.g. 0_0 -> 125_0
#            dataset = []
#            dataset_buffer = []
#            para_old = ""
#            for i in range(len(image_path)):
#                if (relative_path[i] != para_old and i != 0) or i == len(image_path)-1:
#                    dataset += [dataset_buffer]
#                    dataset_buffer = []
#                    
#                dataset_buffer += [[image_path[i], relative_path[i]]]
#                para_old = relative_path[i]
#                
#            del dataset_buffer
#            
#            # d example
#            # d = [[image_path_layer0000, relative_path_layer0000],
#            #      [image_path_layer0010, relative_path_layer0010]]
#            for d in dataset:
#                # split image_path into two lists for training and validation
#                d_1, d_2, r_element = toolbox.split_list_randome(d[1:],
#                                                                 percentage=90)
#                
#                # load data
#                prefix = d[0][1] + "_"
#                
#                nn.load_ground_truth_data([d[0][0]],
#                                          prefix=prefix,
#                                          resize=resize,
#                                          x_pixel=pixel, y_pixel=pixel)
#                nn.load_validation_ground_truth_data([d[0][0]],
#                                                     prefix=prefix,
#                                                     resize=resize,
#                                                     x_pixel=pixel, y_pixel=pixel)
#                
#                ground_truth_filename = prefix + toolbox.get_file_name(d[0][0])
#                for data in d_1:
#                    nn.load_training_data([data[0]],
#                                          ground_truth_filename,
#                                          prefix = prefix,
#                                          resize=resize,
#                                          x_pixel=pixel, y_pixel=pixel)
#                for data in d_2:
#                    nn.load_validation_data([data[0]],
#                                            ground_truth_filename,
#                                            prefix = prefix,
#                                            resize=resize,
#                                            x_pixel=pixel, y_pixel=pixel)
        
        # ---------------------------------------------
        # NEURONAL NETWORK: train network
        # ---------------------------------------------
        toolbox.print_program_section_name("NEURONAL NETWORK: train network")
        
        def process(training_path, ground_truth_path):
            train = ti.get_image_as_npy(training_path)
            ground_truth = ti.get_image_as_npy(ground_truth_path)
        
            return train, ground_truth
        
        batch_size = 16
        fit_epochs = 200
        optimizer = m.get_optimizer([fit_epochs])
#        optimizer = m.get_optimizer()
        nn.train_network([], [],
                         'sparse_categorical_crossentropy', optimizer,
                         fit_epochs=fit_epochs, fit_batch_size=batch_size,
                         process_data=process,
                         filter_layer_number=[[1,20]])
##        nn.train_network(training_data, ground_truth,
##                         'sparse_categorical_crossentropy', optimizer,
##                         fit_epochs=fit_epochs, fit_batch_size=batch_size)
        
        # ---------------------------------------------
        # NEURONAL NETWORK: validata network
        # ---------------------------------------------
        toolbox.print_program_section_name("NEURONAL NETWORK: validate network")
    
#        path = nn.path_intermediate_data_trained_weights + "/weights.hdf5"
#        nn.validate_network(trained_weights_path=path)
        nn.validate_network(pred_invert_colors=True)
        
        # ---------------------------------------------
        # NEURONAL NETWORK: Evaluation
        # ---------------------------------------------
        image_ground_truth_path, image_prediction_path = nn.get_validation_file_paths()
        nn.evaluate_network([nn.jaccard_index, nn.pearson_correlation_coefficient],
                            image_ground_truth_path,
                            image_prediction_path)

        # ---------------------------------------------
        # NEURONAL NETWORK: test network
        # ---------------------------------------------
    #    toolbox.print_program_section_name("NEURONAL NETWORK: test network")
    
    #    nn.test_network(image_test_speckle_path, model)
        
        
    def run(self, external_module_paths, external_module_label):
        self.nn_main(external_module_paths, external_module_label)