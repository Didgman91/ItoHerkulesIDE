#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:22:54 2018

@author: maxdh
"""

import os

import time

from lib.f2 import f2
from lib.toolbox import toolbox
from lib.toolbox.zipData import zip_data
from lib.toolbox import images as ti
from lib.neuronal_network import neuronal_network

from config.neuronal_network.model.model import model_deep_specle_correlation_class

# By calling a module, e.g. F2_main(), the folder name is written to this list.
# This folder is located in the "data" folder and contains all data of the
# module. This will be zipped later.
executed_modules = []

def f2_main(folder, shift):
    global executed_modules
    executed_modules += ["f2"]
    # -------------------------------------
    # F2
    # -------------------------------------
    toolbox.print_program_section_name("F2")

    number_of_layers = 2
    distance = 2  # [m]

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

#    scatter_plate_random = f2.create_scatter_plate(number_of_layers, distance)
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


def nn_main(layer, neuronal_network_path_extension_pretrained_weights=""):
    global executed_modules
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
        fit_epochs=1, fit_batch_size=batch_size,
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


def nn_all_layers(layers, path_extension=""):
    global executed_modules
    executed_modules += ["neuronal_network"]
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
        "data/20181209_F2/input/NIST", ["bmp"])
    image_ground_truth_path = nn.load_ground_truth_data(
        image_path[:-10], layers)
    image_validation_ground_truth_path = nn.load_validation_ground_truth_data(
        image_path[-10:], layers)

    image_speckle_path = []
    image_validation_speckle_path = []
    for layer in layers:
        image_path = toolbox.get_file_path_with_extension(
            "data/20181209_F2/output/speckle/"+layer, ["bmp"])

        image_speckle_path += nn.load_training_data(image_path[:-10], layer)
        image_validation_speckle_path += nn.load_validation_data(image_path[-10:], layer)
    
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
    nn.train_network(
        image_speckle_path, image_ground_truth_path,
        'sparse_categorical_crossentropy', optimizer,
        fit_epochs=1, fit_batch_size=batch_size,
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
                        image_validation_speckle_path,
                        path_pred)

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


#folder, path, layer = f2_main("", 0)
#folder, path, layer = f2_main("", 1000)
for i in range(1,6):
    folder, path, layer = f2_main("", pow(10,i))
    
#    # ---------------------------------------------
#    # CALCULATE MEMORY EFFECT
#    # ---------------------------------------------
#    executed_modules += "memory_effect"
#    toolbox.print_program_section_name("CALCULATE MEMORY EFFECT")
#    
#    path_memory_effect = "data/memory_effect/input/variable_fog"
#    os.makedirs(path_memory_effect, 0o777, True)
#    
#    path_speckle = path_memory_effect + "/{},0".format(pow(10,i))
#    
#    os.makedirs(path_speckle, 0o777, True)
#    
#    for f in folder:
#        if f[-1] == "/":
#            f = f[:-1]
#        destination_folder_name = os.path.split(f)[-1]
#        toolbox.copy_folder(f, path_speckle + "/" + destination_folder_name)
    
# ---------------------------------------------
# BACKUP AND DEDUPLICATION BY COMPRESSION
# ---------------------------------------------
toolbox.print_program_section_name("BACKUP AND DEDUPLICATION BY COMPRESSION")

path_backup = "backup"
os.makedirs(path_backup, 0o777, True)

modules = ""
folders = []
for m in executed_modules:
    modules += "_" + m
    folders += ["data/" + m]

time.strftime("%y%m%d_%H%M")
t = time.strftime("%y%m%d_%H%M")
zip_settings = {'zip_file_name': "{}/{}{}.zip".format(path_backup, t, modules),
                'zip_include_folder_list': ["config", "lib"] + folders,
                'zip_include_file_list': ["main.py"],
                'skipped_folders': [".git", "__pycache__"]}

zip_data(zip_settings)

executed_modules = []

#toolbox.send_message("main.py finished")
