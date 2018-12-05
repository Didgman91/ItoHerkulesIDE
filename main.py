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
    
    numberOfLayers = 100
    
    F2.generate_folder_structure()
    
    # -------------------------------------
    # IMPORT: Image
    # -------------------------------------
    #toolbox.print_program_section_name("IMPORT: MNIST images")
    #      
    #imagePath = F2.load_MNIST_train_images("data/imagePool/MNIST/samples", range(10))
    
    
    toolbox.print_program_section_name("IMPORT: NIST images")
    
    #imagePath = toolbox.get_file_path_with_extension("../../imagePool/NIST/by_write/hsf_0/f0000_14/c0000_14/", ["png"])
    imagePath = toolbox.get_file_path_with_extension_include_subfolders(folder, ["png"])
    imagePath = F2.load_image(imagePath, True, True, 64, 64)
    
    if imagePath == []:
        print("no files")
        exit()
    
    # ---------------------------------------------
    # F2: Generate and save scatter plate
    # ---------------------------------------------
    toolbox.print_program_section_name("F2: Generate and save scatter plate")
          
    scatterPlateRandom = F2.create_scatter_plate(numberOfLayers)
    #scatterPlateRandom = ['data/F2/intermediateData/scatterPlate/ScatterPlateRandomX', 'data/F2/intermediateData/scatterPlate/ScatterPlateRandomY']
    
    # -------------------------------------------------
    # F2: Load scatter plate and calculate speckle
    # -------------------------------------------------
    toolbox.print_program_section_name("F2: Load scatter plate and calculate speckle")
          
    
    imagePath = toolbox.get_file_path_with_extension("data/F2/input/NIST/", ["bmp"])
            
    F2.calculate_propagation(imagePath, scatterPlateRandom, numberOfLayers)
    
    folder, path, layer = F2.sortToFolderByLayer()
    
    #import matplotlib.pyplot as plt
    #
    #for i in range(len(images)):
    #    plt.figure()
    #    plt.imshow(images[i], cmap='gray')
    #    plt.axis('off')
    #    plt.title(imagePath[i])
    #    
    #    
    #    plt.show()
    
    return folder, path, layer


def NN(layer, neuronalNetworkPathExtensionPretrainedWeights=""):
    
    nn = neuronalNetwork.neuronalNetworkClass(layer)
    
    # ---------------------------------------------
    # NEURONAL NETWORK: load data
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: load data")
    
    print("the model is loading...")
    modelFilePath = "lib/neuronalNetwork/model.py"
    model = nn.loadModel(modelFilePath, neuronalNetworkPathExtensionPretrainedWeights)
    print("done")
    
    print("the training and test datasets are loading...")
    imagePath = toolbox.get_file_path_with_extension("data/20181130_F2/input/NIST", ["bmp"])
    imageGroundTruthPath = nn.loadGroundTruthData(imagePath[:-10])
    imageTestGroundTruthPath = nn.loadTestGroundTruthData(imagePath[-10:])
    
    
    imagePath = toolbox.get_file_path_with_extension("data/20181130_F2/output/speckle/"+layer, ["bmp"])
    imageSpecklePath = nn.loadTrainingData(imagePath[:-10])
    imageTestSpecklePath = nn.loadTestData(imagePath[-10:])
    print("done")
    
    
    
    
    # ---------------------------------------------
    # NEURONAL NETWORK: train network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: train network")
    
    model = nn.trainNetwork(imageSpecklePath, imageGroundTruthPath, model, fitEpochs=100, fitBatchSize=16)
    
    
    # ---------------------------------------------
    # NEURONAL NETWORK: test network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: test network")
    
    nn.testNetwork(imageTestSpecklePath, model)
    
    # ---------------------------------------------
    # NEURONAL NETWORK: Evaluation
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: Evaluation")

def NnAllLayers(layers):
    
    nn = neuronalNetwork.neuronalNetworkClass()
    
    # ---------------------------------------------
    # NEURONAL NETWORK: load data
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: load data")
    
    print("the model is loading...")
    modelFilePath = "lib/neuronalNetwork/model.py"
    model = nn.loadModel(modelFilePath)
    model = nn.load_Weights()
    print("done")
    
    print("the training and test datasets are loading...")
    imagePath = toolbox.get_file_path_with_extension("data/F2/input/NIST", ["bmp"])
#    imageGroundTruthPath = nn.loadGroundTruthData(imagePath[:-10], layers)        
    imageTestGroundTruthPath = nn.loadTestGroundTruthData(imagePath[-10:], layers)
    
    imageSpecklePath = []
    imageTestSpecklePath = []
    for layer in layers:
        imagePath = toolbox.get_file_path_with_extension("data/F2/output/speckle/"+layer, ["bmp"])
        
#        imageSpecklePath += nn.loadTrainingData(imagePath[:-10], layer)
        imageTestSpecklePath += nn.loadTestData(imagePath[-10:], layer)
    print("done")
    
    
    
    
#    # ---------------------------------------------
#    # NEURONAL NETWORK: train network
#    # ---------------------------------------------
#    toolbox.print_program_section_name("NEURONAL NETWORK: train network")
#    
#    model = nn.trainNetwork(imageSpecklePath, imageGroundTruthPath, model, fitEpochs=100, fitBatchSize=16)
    
    # ---------------------------------------------
    # NEURONAL NETWORK: test network
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: test network")
    
    nn.testNetwork(imageTestSpecklePath, model)
    
    # ---------------------------------------------
    # NEURONAL NETWORK: Evaluation
    # ---------------------------------------------
    toolbox.print_program_section_name("NEURONAL NETWORK: Evaluation")
    
    #nn.evaluate_Network()

#folder, path, layer = F2_main("../../imagePool/NIST/by_write/hsf_0")

dirs = os.listdir("data/F2/output/speckle/")
dirs.sort()
#NN(dirs[-1])

NnAllLayers(dirs)

toolbox.send_Message("main.py finished")

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
