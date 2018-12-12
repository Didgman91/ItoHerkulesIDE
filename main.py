#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:22:54 2018

@author: maxdh
"""

import os

import time

from lib.F2 import F2
from lib.toolbox import toolbox
from lib.toolbox.zipData import zip_Data
from lib.neuronalNetwork import neuronalNetwork

from config.neuronalNetwork.model.model import model_Deep_Specle_Correlation_Class

# By calling a module, e.g. F2_main(), the folder name is written to this list.
# This folder is located in the "data" folder and contains all data of the
# module. This will be zipped later.
executed_Modules = []

def F2_main(folder):
    global executed_Modules
    executed_Modules += ["F2"]
    # -------------------------------------
    # F2
    # -------------------------------------
    toolbox.print_program_section_name("F2")

    number_Of_Layers = 100
    distance = 100  # [m]

    F2.generate_folder_structure()

    # -------------------------------------
    # IMPORT: Image
    # -------------------------------------
    #toolbox.print_program_section_name("IMPORT: MNIST images")
    #
    #image_Path = F2.load_MNIST_train_images("data/imagePool/MNIST/samples", range(10))

    toolbox.print_program_section_name("IMPORT: NIST images")

#    image_Path = toolbox.get_file_path_with_extension("../../imagePool/NIST/by_write/hsf_0/f0000_14/c0000_14/", ["png"])
#    image_Path = toolbox.get_file_path_with_extension_include_subfolders(folder,
#                                                                         ["png"])
    image_Path = ["rect815.png"]
    image_Path = F2.load_image(image_Path[:1], invertColor=True, resize=True, xPixel=64, yPixel=64)

    if image_Path == []:
        print("no files")
        exit()

    # ---------------------------------------------
    # F2: Generate and save scatter plate
    # ---------------------------------------------
    toolbox.print_program_section_name("F2: Generate and save scatter plate")

    scatter_Plate_Random = F2.create_scatter_plate(number_Of_Layers, distance)
    #scatter_Plate_Random = ['data/F2/intermediateData/scatterPlate/scatter_Plate_RandomX', 'data/F2/intermediateData/scatterPlate/scatter_Plate_RandomY']

    # -------------------------------------------------
    # F2: Load scatter plate and calculate speckle
    # -------------------------------------------------
    toolbox.print_program_section_name(
        "F2: Load scatter plate and calculate speckle")

    image_Path = toolbox.get_file_path_with_extension(
        "data/F2/input/NIST/", ["bmp"])

    F2.calculate_propagation(
        image_Path, scatter_Plate_Random, number_Of_Layers, distance)

    folder, path, layer = F2.sortToFolderByLayer()

    #import matplotlib.pyplot as plt
    #
    # for i in range(len(images)):
    #    plt.figure()
    #    plt.imshow(images[i], cmap='gray')
    #    plt.axis('off')
    #    plt.title(image_Path[i])
    #
    #
    #    plt.show()

    return folder, path, layer


def NN(layer, neuronal_Network_Path_Extension_Pretrained_Weights=""):
    global executed_Modules
    executed_Modules += ["neuronalNetwork"]
    # ---------------------------------------------
    # NEURONAL NETWORK: load data
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: load data")

    print("the model is loading...")
    m = model_Deep_Specle_Correlation_Class()

    nn = neuronalNetwork.neuronal_Network_Class(m.get_Model(), layer)
#    nn = neuronalNetwork.neuronal_Network_Class(layer)
#    model_File_Path = "lib/neuronalNetwork/model.py"
#    model = nn.load_Model(
#        model_File_Path, neuronal_Network_Path_Extension_Pretrained_Weights)
    print("done")

    print("the training and test datasets are loading...")
    image_Path = toolbox.get_file_path_with_extension(
        "data/F2/input/NIST", ["bmp"])
    image_Ground_Truth_Path = nn.load_Ground_Truth_Data(image_Path[:-10])
    image_Validation_Ground_Truth_Path = nn.load_Validation_Ground_Truth_Data(
        image_Path[-10:])

    image_Path = toolbox.get_file_path_with_extension(
        "data/F2/output/speckle/"+layer, ["bmp"])
    image_Speckle_Path = nn.load_Training_Data(image_Path[:-10])
    image_Validation_Speckle_Path = nn.load_Validation_Data(image_Path[-10:])
    print("done")

    # ---------------------------------------------
    # NEURONAL NETWORK: train network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: train network")

    nn.train_Network(
        image_Speckle_Path, image_Ground_Truth_Path,
        fit_Epochs=100, fit_Batch_Size=16)

    # ---------------------------------------------
    # NEURONAL NETWORK: validata network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: validate network")

    nn.validate_Network(image_Validation_Speckle_Path)

    # ---------------------------------------------
    # NEURONAL NETWORK: test network
    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: test network")

#    nn.test_Network(image_Test_Speckle_Path, model)

    # ---------------------------------------------
    # NEURONAL NETWORK: Evaluation
    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: Evaluation")


def NnAllLayers(layers, path_Extension=""):
#    global executed_Modules
#    executed_Modules += ["neuronalNetwork"]
    # ---------------------------------------------
    # NEURONAL NETWORK: load data
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: load data")

    print("the model is loading...")
    m = model_Deep_Specle_Correlation_Class()

    nn = neuronalNetwork.neuronal_Network_Class(m.get_Model(), path_Extension)
#    model = nn.load_Weights()
    print("done")

    print("the training and test datasets are loading...")
    image_Path = toolbox.get_file_path_with_extension(
        "data/20181209_F2/input/NIST", ["bmp"])
    image_Ground_Truth_Path = nn.load_Ground_Truth_Data(
        image_Path[:-10], layers)
    image_Validation_Ground_Truth_Path = nn.load_Validation_Ground_Truth_Data(
        image_Path[-10:], layers)

    image_Speckle_Path = []
    image_Validation_Speckle_Path = []
    for layer in layers:
        image_Path = toolbox.get_file_path_with_extension(
            "data/20181209_F2/output/speckle/"+layer, ["bmp"])

        image_Speckle_Path += nn.load_Training_Data(image_Path[:-10], layer)
        image_Validation_Speckle_Path += nn.load_Validation_Data(image_Path[-10:], layer)
    print("done")

    # ---------------------------------------------
    # NEURONAL NETWORK: train network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: train network")

    nn.train_Network(
        image_Speckle_Path, image_Ground_Truth_Path,
        fit_Epochs=1, fit_Batch_Size=16)

    # ---------------------------------------------
    # NEURONAL NETWORK: validata network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: validate network")

    path_pred = nn.validate_Network(image_Validation_Speckle_Path)
x
    # ---------------------------------------------
    # NEURONAL NETWORK: test network
    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: test network")

#    nn.test_Network(image_Test_Speckle_Path, model)

    # ---------------------------------------------
    # NEURONAL NETWORK: Evaluation
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: Evaluation")
    
    nn.evaluate_Network([nn.jaccard_index, nn.pearson_correlation_coefficient],
                        image_Validation_Speckle_Path,
                        path_pred)

#folder, path, layer = F2_main("../imagePool/NIST/by_write/hsf_0/f0000_14/c0000_14/")

dirs = os.listdir("data/20181209_F2/output/speckle/")
dirs.sort()
#for i in range(len(dirs)):
#    if i % 5 == 0:
#        toolbox.print_program_section_name("DIRECTORY: {}".format(dirs[i]))
#        NN(dirs[i])

toolbox.print_program_section_name("NN All: first 3 m")
NnAllLayers(dirs[:3], "3meter")

#toolbox.print_program_section_name("NN All: first 10 m")
#NnAllLayers(dirs[:10], "10meter")

#toolbox.print_program_section_name("NN All: first 20 m")
#NnAllLayers(dirs[:20], "20meter")

#toolbox.print_program_section_name("NN All: first 40 m")
#NnAllLayers(dirs[:40], "40meter")

#toolbox.print_program_section_name("NN All: first 60 m")
#NnAllLayers(dirs[:60], "60meter")

#toolbox.print_program_section_name("NN All: first 80 m")
#NnAllLayers(dirs[:80], "80meter")

#toolbox.print_program_section_name("NN All: first 100 m")
#NnAllLayers(dirs, "100meter")

# for i in range(len(dirs)):
#    if i % 5 == 0:
#        toolbox.print_program_section_name("DIRECTORY: {}".format(dirs[i]))
#        NN(dirs[i])
# if i==0:
# NN(dirs[i])
# else:
##            NN(dirs[i], dirs[i-1])
# if i>1:
# break

# -------------------------------------
# COPY FILES TO SERVER
# -------------------------------------
#
#
# -------------------------------------
# CACLULATE SPECKLE
# -------------------------------------
#
#output, exitCode, t = F2.run_Process("ls", "-lsa")
#
# for i in range(len(output)):
#    print("")
#    print(output[i])
#
#
#print("time [s]: %.6f" % (t))
#
#


# ---------------------------------------------
# BACKUP AND DEDUPLICATION BY COMPRESSION
# ---------------------------------------------
toolbox.print_program_section_name("BACKUP AND DEDUPLICATION BY COMPRESSION")

path_Backup = "backup"
os.makedirs(path_Backup, 0o777, True)

modules = ""
folders = []
for m in executed_Modules:
    modules += "_" + m
    folders += ["data/" + m]

time.strftime("%y%m%d_%H%M")
t = time.strftime("%y%m%d_%H%M")
zip_Settings = {'zip_File_Name': "{}/{}{}.zip".format(path_Backup, t, modules),
                'zip_Include_Folder_List': ["config", "lib"] + folders,
                'zip_Include_File_List': ["main.py"],
                'skipped_Folders': [".git", "__pycache__"]}

zip_Data(zip_Settings)

#toolbox.send_Message("main.py finished")
