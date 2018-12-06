#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:22:54 2018

@author: maxdh
"""

import os

from lib.F2 import F2
from lib.toolbox import toolbox
from lib.neuronalNetwork import neuronalNetwork


def F2_main(folder):
    # -------------------------------------
    # F2
    # -------------------------------------
    toolbox.print_program_section_name("F2")
    
    number_Of_Layers = 100
    
    F2.generate_folder_structure()
    
    # -------------------------------------
    # IMPORT: Image
    # -------------------------------------
    #toolbox.print_program_section_name("IMPORT: MNIST images")
    #      
    #image_Path = F2.load_MNIST_train_images("data/imagePool/MNIST/samples", range(10))
    
    
    toolbox.print_program_section_name("IMPORT: NIST images")
    
    #image_Path = toolbox.get_file_path_with_extension("../../imagePool/NIST/by_write/hsf_0/f0000_14/c0000_14/", ["png"])
    image_Path = toolbox.get_file_path_with_extension_include_subfolders(folder, ["png"])
    image_Path = F2.load_image(image_Path, True, True, 64, 64)
    
    if image_Path == []:
        print("no files")
        exit()
    
    # ---------------------------------------------
    # F2: Generate and save scatter plate
    # ---------------------------------------------
    toolbox.print_program_section_name("F2: Generate and save scatter plate")
          
    scatter_Plate_Random = F2.create_scatter_plate(number_Of_Layers)
    #scatter_Plate_Random = ['data/F2/intermediateData/scatterPlate/scatter_Plate_RandomX', 'data/F2/intermediateData/scatterPlate/scatter_Plate_RandomY']
    
    # -------------------------------------------------
    # F2: Load scatter plate and calculate speckle
    # -------------------------------------------------
    toolbox.print_program_section_name("F2: Load scatter plate and calculate speckle")
          
    
    image_Path = toolbox.get_file_path_with_extension("data/F2/input/NIST/", ["bmp"])
            
    F2.calculate_propagation(image_Path, scatter_Plate_Random, number_Of_Layers)
    
    folder, path, layer = F2.sortToFolderByLayer()
    
    #import matplotlib.pyplot as plt
    #
    #for i in range(len(images)):
    #    plt.figure()
    #    plt.imshow(images[i], cmap='gray')
    #    plt.axis('off')
    #    plt.title(image_Path[i])
    #    
    #    
    #    plt.show()
    
    return folder, path, layer


def NN(layer, neuronal_Network_Path_Extension_Pretrained_Weights=""):
    
    nn = neuronalNetwork.neuronal_Network_Class(layer)
    
    # ---------------------------------------------
    # NEURONAL NETWORK: load data
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: load data")
    
    print("the model is loading...")
    model_File_Path = "lib/neuronalNetwork/model.py"
    model = nn.load_Model(model_File_Path, neuronal_Network_Path_Extension_Pretrained_Weights)
    print("done")
    
    print("the training and test datasets are loading...")
    image_Path = toolbox.get_file_path_with_extension("data/20181130_F2/input/NIST", ["bmp"])
    image_Ground_Truth_Path = nn.load_Ground_Truth_Data(image_Path[:-10])
    image_Test_Ground_Truth_Path = nn.load_Test_Ground_Truth_Data(image_Path[-10:])
    
    
    image_Path = toolbox.get_file_path_with_extension("data/20181130_F2/output/speckle/"+layer, ["bmp"])
    image_Speckle_Path = nn.load_Training_Data(image_Path[:-10])
    image_Test_Speckle_Path = nn.load_Test_Data(image_Path[-10:])
    print("done")
    
    
    
    
    # ---------------------------------------------
    # NEURONAL NETWORK: train network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: train network")
    
    model = nn.train_Network(image_Speckle_Path, image_Ground_Truth_Path, model, fit_Epochs=100, fit_Batch_Size=16)
    
    
    # ---------------------------------------------
    # NEURONAL NETWORK: test network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: test network")
    
    nn.test_Network(image_Test_Speckle_Path, model)
    
    # ---------------------------------------------
    # NEURONAL NETWORK: Evaluation
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: Evaluation")

def NnAllLayers(layers):
    
    nn = neuronalNetwork.neuronal_Network_Class()
    
    # ---------------------------------------------
    # NEURONAL NETWORK: load data
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: load data")
    
    print("the model is loading...")
    model_File_Path = "lib/neuronalNetwork/model.py"
    model = nn.load_Model(model_File_Path)
#    model = nn.load_Weights()
    print("done")
    
    print("the training and test datasets are loading...")
    image_Path = toolbox.get_file_path_with_extension("data/20181130_F2/input/NIST", ["bmp"])
    image_Ground_Truth_Path = nn.load_Ground_Truth_Data(image_Path[:-10], layers)
    image_Test_Ground_Truth_Path = nn.load_Test_Ground_Truth_Data(image_Path[-10:], layers)
    
    image_Speckle_Path = []
    image_Test_Speckle_Path = []
    for layer in layers:
        image_Path = toolbox.get_file_path_with_extension("data/20181130_F2/output/speckle/"+layer, ["bmp"])
        
        image_Speckle_Path += nn.load_Training_Data(image_Path[:-10], layer)
        image_Test_Speckle_Path += nn.load_Test_Data(image_Path[-10:], layer)
    print("done")
    
    
    
    
    # ---------------------------------------------
    # NEURONAL NETWORK: train network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: train network")
    
    model = nn.train_Network(image_Speckle_Path, image_Ground_Truth_Path, model, fit_Epochs=10, fit_Batch_Size=16)
    
    # ---------------------------------------------
    # NEURONAL NETWORK: test network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: test network")
    
    nn.test_Network(image_Test_Speckle_Path, model)
    
    # ---------------------------------------------
    # NEURONAL NETWORK: Evaluation
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: Evaluation")
    
    nn.evaluate_Network(jd)

#folder, path, layer = F2_main("../../imagePool/NIST/by_write/hsf_0")

dirs = os.listdir("data/F2/output/speckle/")
dirs.sort()
#NN(dirs[-1])

NnAllLayers(dirs[:1])

#for i in range(len(dirs)):
#    if i % 5 == 0:
#        toolbox.print_program_section_name("DIRECTORY: {}".format(dirs[i]))
#        NN(dirs[i])
##        if i==0:
##            NN(dirs[i])
##        else:
##            NN(dirs[i], dirs[i-1])
##    if i>1:
##        break

## -------------------------------------
## COPY FILES TO SERVER
## -------------------------------------
#
#
## -------------------------------------
## CACLULATE SPECKLE
## -------------------------------------
#
#output, exitCode, t = F2.run_Process("ls", "-lsa")
#
#for i in range(len(output)):
#    print("")
#    print(output[i])
#    
#
#print("time [s]: %.6f" % (t))
#
#

#toolbox.send_Message("main.py finished")
