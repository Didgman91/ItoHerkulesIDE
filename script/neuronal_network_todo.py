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

def init():
#    neuronal_network.set_tf_gpu_fraction()
    neuronal_network.set_tf_gpu_allow_growth()

def nn_main(layer, neuronal_network_path_extension_pretrained_weights=""):
#    global executed_modules
    executed_modules += ["neuronal_network"]
    # ---------------------------------------------
    # NEURONAL NETWORK: load data
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: load data")

    print("the model is loading...")
    m = model_deep_specle_correlation_class()

    nn = neuronal_network.neuronal_network_class(m.get_model(), layer)

    print("done")

    print("the training and test datasets are loading...")
    image_path = toolbox.get_file_path_with_extension(
        "data/f2/input/nist", ["bmp"])
    image_ground_truth_path = nn.load_ground_truth_data(image_path[:-10])
    image_validation_ground_truth_path = nn.load_validation_ground_truth_data(
        image_path[-10:])

    image_path = toolbox.get_file_path_with_extension(
        "data/f2/output/speckle/"+layer, ["bmp"])
    image_speckle_path = nn.load_training_data(image_path[:-10])
    image_validation_speckle_path = nn.load_validation_data(image_path[-10:])
    print("done")

    # ---------------------------------------------
    # NEURONAL NETWORK: train network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: train network")

    image_speckle_path.sort()
    image_ground_truth_path.sort()
    
#    training_data = ti.get_image_as_npy(image_speckle_path)
#    ground_truth = ti.get_image_as_npy(image_ground_truth_path)
    
    def process(training_path, ground_truth_path):
        train = ti.get_image_as_npy(training_path)
        ground_truth = ti.get_image_as_npy(ground_truth_path)
    
        return train, ground_truth
    
    batch_size = 16
    optimizer = m.get_optimizer([batch_size])
    nn.train_network(
        image_speckle_path, image_ground_truth_path,
        'sparse_categorical_crossentropy', optimizer,
        fit_epochs=100, fit_batch_size=batch_size,
        process_data=process)

    # ---------------------------------------------
    # NEURONAL NETWORK: validata network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: validate network")

    image_validation_speckle_path.sort()
    data = ti.get_image_as_npy(image_validation_speckle_path)

    fileName = []
    for ip in image_validation_speckle_path:
        base = os.path.basename(ip)
        fileName += [os.path.splitext(base)[0]]

    pred = nn.validate_network(data)

    path_pred = ti.save_4D_npy_as_bmp(pred, fileName, nn.path_data +
                                      nn.path_output_validation_data_prediction,
                                      invert_color=True)

    # ---------------------------------------------
    # NEURONAL NETWORK: test network
    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: test network")

#    nn.test_network(image_test_speckle_path, model)

    # ---------------------------------------------
    # NEURONAL NETWORK: Evaluation
    # ---------------------------------------------
    nn.evaluate_network([nn.jaccard_index, nn.pearson_correlation_coefficient],
                        image_validation_speckle_path,
                        path_pred)

def nn_all_layers(path_extension=""):
    layers = os.listdir("data/f2/output/speckle/")
    
#    global executed_modules
#    executed_modules += ["neuronal_network"]
    # ---------------------------------------------
    # NEURONAL NETWORK: load data
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: load data")

    print("the model is loading...")
    m = model_deep_specle_correlation_class()

    nn = neuronal_network.neuronal_network_class(m.get_model(), path_extension)
#    model = nn.load_weights()
    print("done")

    print("the training and test datasets are loading...")
    image_path = toolbox.get_file_path_with_extension(
        "data/f2/input/nist", ["bmp"])
    # only intesity images
#    image_path = [s for s in image_path if "Intensity" in s]
    image_path.sort();
    
    image_ground_truth_path = nn.load_ground_truth_data(
        image_path[:-10], layers, resize=True)
    image_validation_ground_truth_path = nn.load_validation_ground_truth_data(
        image_path[-10:], layers, resize=True)

    image_speckle_path = []
    image_validation_speckle_path = []
    for layer in layers:
        image_path = toolbox.get_file_path_with_extension(
            "data/f2/output/speckle/"+layer, ["bmp"])
        
        # only intesity images
        image_path = [s for s in image_path if "Intensity" in s]
        image_path.sort()

        image_speckle_path += nn.load_training_data(image_path[:-10], resize=True)
        image_validation_speckle_path += nn.load_validation_data(image_path[-10:], resize=True)
    
    image_speckle_path.sort()
    image_validation_speckle_path.sort()
    print("done")

    # ---------------------------------------------
    # NEURONAL NETWORK: train network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: train network")
    
    image_speckle_path.sort()
    image_ground_truth_path.sort()
    
#    training_data = ti.get_image_as_npy(image_speckle_path)
#    ground_truth = ti.get_image_as_npy(image_ground_truth_path)
    
    def process(training_path, ground_truth_path):
        train = ti.get_image_as_npy(training_path)
        ground_truth = ti.get_image_as_npy(ground_truth_path)
    
        return train, ground_truth
    
    batch_size = 16
    optimizer = m.get_optimizer([batch_size])
#    nn.train_network(
#        training_data, ground_truth,
#        'sparse_categorical_crossentropy', optimizer,
#        fit_epochs=100, fit_batch_size=batch_size,
#        process_data=[])
    nn.train_network(
        image_speckle_path, image_ground_truth_path,
        'sparse_categorical_crossentropy', optimizer,
        fit_epochs=100, fit_batch_size=batch_size,
        process_data=process)

    # ---------------------------------------------
    # NEURONAL NETWORK: validata network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: validate network")

    image_validation_speckle_path.sort()
    data = ti.get_image_as_npy(image_validation_speckle_path)

    fileName = []
    for ip in image_validation_speckle_path:
        base = os.path.basename(ip)
        fileName += [os.path.splitext(base)[0]]

    pred, path = nn.validate_network(data)

    path_pred = ti.save_4D_npy_as_bmp(pred, fileName, path,
                                      invert_color=True)

    # ---------------------------------------------
    # NEURONAL NETWORK: test network
    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: test network")

#    nn.test_network(image_test_speckle_path, model)

    # ---------------------------------------------
    # NEURONAL NETWORK: Evaluation
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: Evaluation")
    
    nn.evaluate_network([nn.jaccard_index, nn.pearson_correlation_coefficient],
                        image_validation_speckle_path, path_pred)

#def nn_all_layers(path_extension=""):
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
#        "data/F2/input/NIST", ["bmp"])
#    
##    ready = toolbox.read_file_lines("script/ready.txt")
##    ready = list(set(ready))
##    image_path = toolbox.get_intersection(image_path, ready, False)
#    
#    image_ground_truth_path = nn.load_ground_truth_data(
#        image_path[:-10])
#    
#    image_validation_ground_truth_path = nn.load_validation_ground_truth_data(
#        image_path[-10:])
#
##    image_speckle_path = []
##    image_validation_speckle_path = []
##    for layer in layers:
#    image_path = toolbox.get_file_path_with_extension(
#        "data/F2/output/speckle/", ["bmp"])
##    ready = toolbox.read_file_lines("script/ready.txt")
##    ready = list(set(ready))
##    image_path = toolbox.get_intersection(image_path, ready, False)
#
#    image_speckle_path = nn.load_training_data(image_path[:-10])
#    image_validation_speckle_path = nn.load_validation_data(image_path[-10:])
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
#    training_data = ti.get_image_as_npy(image_speckle_path)
#    ground_truth = ti.get_image_as_npy(image_ground_truth_path)
#    
#    def process(training_path, ground_truth_path):
#        print("using fit generator: call process()")
#        train = ti.get_image_as_npy(training_path)
#        ground_truth = ti.get_image_as_npy(ground_truth_path)
#    
#        return train, ground_truth
#    
#    batch_size = 16
#    optimizer = m.get_optimizer([batch_size])
#    nn.train_network(
#        training_data, ground_truth,
#        'sparse_categorical_crossentropy', optimizer,
#        fit_epochs=100, fit_batch_size=batch_size,
#        process_data=[])
##    nn.train_network(
##        image_speckle_path, image_ground_truth_path,
##        'sparse_categorical_crossentropy', optimizer,
##        fit_epochs=100, fit_batch_size=batch_size,
##        process_data=process)
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
#                        image_validation_speckle_path,
#                        path_pred)

#folder, path, layer = f2_main("", 10)

#dirs = os.listdir("data/20181209_F2/output/speckle/")
#dirs.sort()
#for i in range(len(dirs)):
#    if i % 5 == 0:
#        toolbox.print_program_section_name("DIRECTORY: {}".format(dirs[i]))
#        nn_main(dirs[i])

#toolbox.print_program_section_name("NN All: first 2 m")
#nn_all_layers(dirs[:2], "2meter")

#toolbox.print_program_section_name("NN All: first 10 m")
#nn_all_layers(dirs[:10], "10meter")

#toolbox.print_program_section_name("NN All: first 20 m")
#nn_all_layers(dirs[:20], "20meter")

#toolbox.print_program_section_name("NN All: first 40 m")
#nn_all_layers(dirs[:40], "40meter")

#toolbox.print_program_section_name("NN All: first 60 m")
#nn_all_layers(dirs[:60], "60meter")

#toolbox.print_program_section_name("NN All: first 80 m")
#nn_all_layers(dirs[:80], "80meter")

#toolbox.print_program_section_name("NN All: first 100 m")
#nn_all_layers(dirs, "100meter")

# for i in range(len(dirs)):
#    if i % 5 == 0:
#        toolbox.print_program_section_name("DIRECTORY: {}".format(dirs[i]))
#        nn_main(dirs[i])
# if i==0:
# nn_main(dirs[i])
# else:
##            NN(dirs[i], dirs[i-1])
# if i>1:
# break
