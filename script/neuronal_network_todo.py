#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:59:04 2019

@author: maxdh
"""
import os


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
        K.set_floatx('float32')
        neuronal_network.set_tf_gpu_fraction(0.24)
#       neuronal_network.set_tf_gpu_allow_growth()


    def nn_main(self, external_module_paths, external_module_label):
        self.nn_init()
        def get_image_paths_from_other_runs(external_folders, external_label):
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
                path_buffer = __filter(path_buffer, "Intensity")
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
    
#        nn = neuronal_network.neuronal_network_class([])
        pixel = 256
        resize = True
        nn = neuronal_network.neuronal_network_class(m.get_model(pixel))
#        nn = neuronal_network.neuronal_network_class([])
#       model = nn.load_weights()
        print("done")
        
#        nn.plot_history()
#        
#        return 
        # nn.load data from file system
        reload_images = False
        if reload_images is True:
            print("the training and test datasets are loading...")
            image_path, relative_path = get_image_paths_from_other_runs(external_module_paths,
                                                                        external_module_label)
            
            for i in range(len(relative_path)):
                relative_path[i] = relative_path[i].replace('/','_')
                relative_path[i] = relative_path[i].replace(',', '_')
                    
            
            
            # split data sets: parameter: relative_path change -> shift change e.g. 0_0 -> 125_0
            dataset = []
            dataset_buffer = []
            para_old = ""
            for i in range(len(image_path)):
                if (relative_path[i] != para_old and i != 0) or i == len(image_path)-1:
                    dataset += [dataset_buffer]
                    dataset_buffer = []
                    
                dataset_buffer += [[image_path[i], relative_path[i]]]
                para_old = relative_path[i]
                
            del dataset_buffer
            
            # d example
            # d = [[image_path_layer0000, relative_path_layer0000],
            #      [image_path_layer0010, relative_path_layer0010]]
            for d in dataset:
                # split image_path into two lists for training and validation
                d_1, d_2, r_element = toolbox.split_list_randome(d[1:],
                                                                 percentage=90)
                
                # load data
                prefix = d[0][1] + "_"
                nn.load_ground_truth_data([d[0][0]],
                                          prefix=prefix,
                                          resize=resize,
                                          x_pixel=pixel, y_pixel=pixel)
                nn.load_validation_ground_truth_data([d[0][0]],
                                                     prefix=prefix,
                                                     resize=resize,
                                                     x_pixel=pixel, y_pixel=pixel)
                
                ground_truth_filename = prefix + toolbox.get_file_name(d[0][0])
                for data in d_1:
                    nn.load_training_data([data[0]],
                                          ground_truth_filename,
                                          prefix = prefix,
                                          resize=resize,
                                          x_pixel=pixel, y_pixel=pixel)
                for data in d_2:
                    nn.load_validation_data([data[0]],
                                            ground_truth_filename,
                                            prefix = prefix,
                                            resize=resize,
                                            x_pixel=pixel, y_pixel=pixel)
        
        # ---------------------------------------------
        # NEURONAL NETWORK: train network
        # ---------------------------------------------
        toolbox.print_program_section_name("NEURONAL NETWORK: train network")
        
        def process(training_path, ground_truth_path):
            train = ti.get_image_as_npy(training_path)
            ground_truth = ti.get_image_as_npy(ground_truth_path)
        
            return train, ground_truth
        
        batch_size = 4
        fit_epochs = 280
        optimizer = m.get_optimizer([fit_epochs])
#        optimizer = m.get_optimizer()
        nn.train_network([], [],
                         'sparse_categorical_crossentropy', optimizer,
                         fit_epochs=fit_epochs, fit_batch_size=batch_size,
                         process_data=process)
#        nn.train_network(training_data, ground_truth,
#                         'sparse_categorical_crossentropy', optimizer,
#                         fit_epochs=fit_epochs, fit_batch_size=batch_size)
        
        # ---------------------------------------------
        # NEURONAL NETWORK: validata network
        # ---------------------------------------------
        toolbox.print_program_section_name("NEURONAL NETWORK: validate network")
    
#        path = nn.path_intermediate_data_trained_weights + "/weights.hdf5"
#        nn.validate_network(trained_weights_path=path)
        nn.validate_network()
        
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

#def nn_main(layer, neuronal_network_path_extension_pretrained_weights=""):
##    global executed_modules
#    executed_modules += ["neuronal_network"]
#    # ---------------------------------------------
#    # NEURONAL NETWORK: load data
#    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: load data")
#
#    print("the model is loading...")
#    m = model_deep_specle_correlation_class()
#
#    nn = neuronal_network.neuronal_network_class(m.get_model(), layer)
#
#    print("done")
#
#    print("the training and test datasets are loading...")
#    image_path = toolbox.get_file_path_with_extension(
#        "data/f2/input/nist", ["bmp"])
#    image_ground_truth_path = nn.load_ground_truth_data(image_path[:-10])
#    image_validation_ground_truth_path = nn.load_validation_ground_truth_data(
#        image_path[-10:])
#
#    image_path = toolbox.get_file_path_with_extension(
#        "data/f2/output/speckle/"+layer, ["bmp"])
#    image_speckle_path = nn.load_training_data(image_path[:-10])
#    image_validation_speckle_path = nn.load_validation_data(image_path[-10:])
#    print("done")
#
#    # ---------------------------------------------
#    # NEURONAL NETWORK: train network
#    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: train network")
#
#    image_speckle_path.sort()
#    image_ground_truth_path.sort()
#    
##    training_data = ti.get_image_as_npy(image_speckle_path)
##    ground_truth = ti.get_image_as_npy(image_ground_truth_path)
#    
#    def process(training_path, ground_truth_path):
#        train = ti.get_image_as_npy(training_path)
#        ground_truth = ti.get_image_as_npy(ground_truth_path)
#    
#        return train, ground_truth
#    
#    batch_size = 16
#    optimizer = m.get_optimizer([batch_size])
#    nn.train_network(
#        image_speckle_path, image_ground_truth_path,
#        'sparse_categorical_crossentropy', optimizer,
#        fit_epochs=100, fit_batch_size=batch_size,
#        process_data=process)
#
#    # ---------------------------------------------
#    # NEURONAL NETWORK: validata network
#    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: validate network")
#
#    image_validation_speckle_path.sort()
#    data = ti.get_image_as_npy(image_validation_speckle_path)
#
#    fileName = []
#    for ip in image_validation_speckle_path:
#        base = os.path.basename(ip)
#        fileName += [os.path.splitext(base)[0]]
#
#    pred = nn.validate_network(data)
#
#    path_pred = ti.save_4D_npy_as_bmp(pred, fileName, nn.path_data +
#                                      nn.path_output_validation_data_prediction,
#                                      invert_color=True)
#
#    # ---------------------------------------------
#    # NEURONAL NETWORK: test network
#    # ---------------------------------------------
##    toolbox.print_program_section_name("NEURONAL NETWORK: test network")
#
##    nn.test_network(image_test_speckle_path, model)
#
#    # ---------------------------------------------
#    # NEURONAL NETWORK: Evaluation
#    # ---------------------------------------------
#    nn.evaluate_network([nn.jaccard_index, nn.pearson_correlation_coefficient],
#                        image_validation_speckle_path,
#                        path_pred)
#
#def nn_all_layers(path_to_layers, path_extension=""):
##    layers = os.listdir("data/f2/output/speckle/")
#    layers = path_to_layers
#    
##    global executed_modules
##    executed_modules += ["neuronal_network"]
#    # ---------------------------------------------
#    # NEURONAL NETWORK: load data
#    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: load data")
#
#    print("the model is loading...")
#    m = model_deep_specle_correlation_class()
#
#    nn = neuronal_network.neuronal_network_class(m.get_model(), path_extension)
##    model = nn.load_weights()
#    print("done")
#
#    print("the training and test datasets are loading...")
#    image_path = toolbox.get_file_path_with_extension(
#        "data/f2/input/nist", ["bmp"])
#    # only intesity images
##    image_path = [s for s in image_path if "Intensity" in s]
#    image_path.sort();
#    
#    image_ground_truth_path = nn.load_ground_truth_data(
#        image_path[:-10], layers, resize=True)
#    image_validation_ground_truth_path = nn.load_validation_ground_truth_data(
#        image_path[-10:], layers, resize=True)
#
#    image_speckle_path = []
#    image_validation_speckle_path = []
#    for layer in layers:
#        image_path = toolbox.get_file_path_with_extension(
#            "data/f2/output/speckle/"+layer, ["bmp"])
#        
#        # only intesity images
#        image_path = [s for s in image_path if "Intensity" in s]
#        image_path.sort()
#
#        image_speckle_path += nn.load_training_data(image_path[:-10], resize=True)
#        image_validation_speckle_path += nn.load_validation_data(image_path[-10:], resize=True)
#    
#    image_speckle_path.sort()
#    image_validation_speckle_path.sort()
#    print("done")
#
#    # ---------------------------------------------
#    # NEURONAL NETWORK: train network
#    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: train network")
#    
#    image_speckle_path.sort()
#    image_ground_truth_path.sort()
#    
##    training_data = ti.get_image_as_npy(image_speckle_path)
##    ground_truth = ti.get_image_as_npy(image_ground_truth_path)
#    
#    def process(training_path, ground_truth_path):
#        train = ti.get_image_as_npy(training_path)
#        ground_truth = ti.get_image_as_npy(ground_truth_path)
#    
#        return train, ground_truth
#    
#    batch_size = 16
#    optimizer = m.get_optimizer([batch_size])
##    nn.train_network(
##        training_data, ground_truth,
##        'sparse_categorical_crossentropy', optimizer,
##        fit_epochs=100, fit_batch_size=batch_size,
##        process_data=[])
#    nn.train_network(
#        image_speckle_path, image_ground_truth_path,
#        'sparse_categorical_crossentropy', optimizer,
#        fit_epochs=100, fit_batch_size=batch_size,
#        process_data=process)
#
#    # ---------------------------------------------
#    # NEURONAL NETWORK: validata network
#    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: validate network")
#
#    image_validation_speckle_path.sort()
#    data = ti.get_image_as_npy(image_validation_speckle_path)
#
#    fileName = []
#    for ip in image_validation_speckle_path:
#        base = os.path.basename(ip)
#        fileName += [os.path.splitext(base)[0]]
#
#    pred, path = nn.validate_network(data)
#
#    path_pred = ti.save_4D_npy_as_bmp(pred, fileName, path,
#                                      invert_color=True)
#
#    # ---------------------------------------------
#    # NEURONAL NETWORK: test network
#    # ---------------------------------------------
##    toolbox.print_program_section_name("NEURONAL NETWORK: test network")
#
##    nn.test_network(image_test_speckle_path, model)
#
#    # ---------------------------------------------
#    # NEURONAL NETWORK: Evaluation
#    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: Evaluation")
#    
#    nn.evaluate_network([nn.jaccard_index, nn.pearson_correlation_coefficient],
#                        image_validation_speckle_path, path_pred)
#
##def nn_all_layers(path_extension=""):
###    global executed_modules
###    executed_modules += ["neuronal_network"]
##    # ---------------------------------------------
##    # NEURONAL NETWORK: load data
##    # ---------------------------------------------
##    toolbox.print_program_section_name("NEURONAL NETWORK: load data")
##
##    print("the model is loading...")
##    m = model_deep_specle_correlation_class()
##
##    nn = neuronal_network.neuronal_network_class(m.get_model(), path_extension)
###    model = nn.load_weights()
##    print("done")
##
##    print("the training and test datasets are loading...")
##    image_path = toolbox.get_file_path_with_extension(
##        "data/F2/input/NIST", ["bmp"])
##    
###    ready = toolbox.read_file_lines("script/ready.txt")
###    ready = list(set(ready))
###    image_path = toolbox.get_intersection(image_path, ready, False)
##    
##    image_ground_truth_path = nn.load_ground_truth_data(
##        image_path[:-10])
##    
##    image_validation_ground_truth_path = nn.load_validation_ground_truth_data(
##        image_path[-10:])
##
###    image_speckle_path = []
###    image_validation_speckle_path = []
###    for layer in layers:
##    image_path = toolbox.get_file_path_with_extension(
##        "data/F2/output/speckle/", ["bmp"])
###    ready = toolbox.read_file_lines("script/ready.txt")
###    ready = list(set(ready))
###    image_path = toolbox.get_intersection(image_path, ready, False)
##
##    image_speckle_path = nn.load_training_data(image_path[:-10])
##    image_validation_speckle_path = nn.load_validation_data(image_path[-10:])
##    
##    image_speckle_path.sort()
##    image_validation_speckle_path.sort()
##    print("done")
##
##    # ---------------------------------------------
##    # NEURONAL NETWORK: train network
##    # ---------------------------------------------
##    toolbox.print_program_section_name("NEURONAL NETWORK: train network")
##    
##    image_speckle_path.sort()
##    image_ground_truth_path.sort()
##    
##    training_data = ti.get_image_as_npy(image_speckle_path)
##    ground_truth = ti.get_image_as_npy(image_ground_truth_path)
##    
##    def process(training_path, ground_truth_path):
##        print("using fit generator: call process()")
##        train = ti.get_image_as_npy(training_path)
##        ground_truth = ti.get_image_as_npy(ground_truth_path)
##    
##        return train, ground_truth
##    
##    batch_size = 16
##    optimizer = m.get_optimizer([batch_size])
##    nn.train_network(
##        training_data, ground_truth,
##        'sparse_categorical_crossentropy', optimizer,
##        fit_epochs=100, fit_batch_size=batch_size,
##        process_data=[])
###    nn.train_network(
###        image_speckle_path, image_ground_truth_path,
###        'sparse_categorical_crossentropy', optimizer,
###        fit_epochs=100, fit_batch_size=batch_size,
###        process_data=process)
##
##    # ---------------------------------------------
##    # NEURONAL NETWORK: validata network
##    # ---------------------------------------------
##    toolbox.print_program_section_name("NEURONAL NETWORK: validate network")
##
##    image_validation_speckle_path.sort()
##    data = ti.get_image_as_npy(image_validation_speckle_path)
##
##    fileName = []
##    for ip in image_validation_speckle_path:
##        base = os.path.basename(ip)
##        fileName += [os.path.splitext(base)[0]]
##
##    pred, path = nn.validate_network(data)
##
##    path_pred = ti.save_4D_npy_as_bmp(pred, fileName, path,
##                                      invert_color=True)
##
##    # ---------------------------------------------
##    # NEURONAL NETWORK: test network
##    # ---------------------------------------------
###    toolbox.print_program_section_name("NEURONAL NETWORK: test network")
##
###    nn.test_network(image_test_speckle_path, model)
##
##    # ---------------------------------------------
##    # NEURONAL NETWORK: Evaluation
##    # ---------------------------------------------
##    toolbox.print_program_section_name("NEURONAL NETWORK: Evaluation")
##    
##    nn.evaluate_network([nn.jaccard_index, nn.pearson_correlation_coefficient],
##                        image_validation_speckle_path,
##                        path_pred)
#
#
#def run():
#    init()
#    
#    path = "../Grid/itodaiber/KW17/" +\
#           "thickSP_500mm_memory_effect_25_mm_fog_d_20_rhon_40_NA_0_12_lam_905_dist_500mm_NAprop_0_01_fog_100m_rhon_0_20/" + \
#           "data/f2/output/speckle/"
#    
#    shifts = [[25,0], [50,0], [125,0], [200,0]]
#    
#    path_to_layers = []
#    for s in range(len(shifts)):
#        buffer = path + "{},{}/".format(s[0], s[1])
#        path_to_layers += [buffer]
#            
#    
#    nn_all_layers(path_to_layers)
#
#
##folder, path, layer = f2_main("", 10)
#
##dirs = os.listdir("data/20181209_F2/output/speckle/")
##dirs.sort()
##for i in range(len(dirs)):
##    if i % 5 == 0:
##        toolbox.print_program_section_name("DIRECTORY: {}".format(dirs[i]))
##        nn_main(dirs[i])
#
##toolbox.print_program_section_name("NN All: first 2 m")
##nn_all_layers(dirs[:2], "2meter")
#
##toolbox.print_program_section_name("NN All: first 10 m")
##nn_all_layers(dirs[:10], "10meter")
#
##toolbox.print_program_section_name("NN All: first 20 m")
##nn_all_layers(dirs[:20], "20meter")
#
##toolbox.print_program_section_name("NN All: first 40 m")
##nn_all_layers(dirs[:40], "40meter")
#
##toolbox.print_program_section_name("NN All: first 60 m")
##nn_all_layers(dirs[:60], "60meter")
#
##toolbox.print_program_section_name("NN All: first 80 m")
##nn_all_layers(dirs[:80], "80meter")
#
##toolbox.print_program_section_name("NN All: first 100 m")
##nn_all_layers(dirs, "100meter")
#
## for i in range(len(dirs)):
##    if i % 5 == 0:
##        toolbox.print_program_section_name("DIRECTORY: {}".format(dirs[i]))
##        nn_main(dirs[i])
## if i==0:
## nn_main(dirs[i])
## else:
###            NN(dirs[i], dirs[i-1])
## if i>1:
## break
