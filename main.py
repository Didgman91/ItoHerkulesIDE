#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:22:54 2018

@author: maxdh
"""

from lib.F2 import F2
from lib.toolbox import toolbox
from lib.neuronalNetwork import neuronalNetwork


F2.generate_folder_structure()

# -------------------------------------
# IMPORT: Image
# -------------------------------------
#toolbox.print_program_section_name("IMPORT: MNIST images")
#      
#imagePath = F2.load_MNIST_train_images("data/imagePool/MNIST/samples", range(10))


toolbox.print_program_section_name("IMPORT: NIST images")

#imagePath = toolbox.get_file_path_with_extension("../../imagePool/NIST/by_write/hsf_0/f0000_14/c0000_14/", ["png"])
imagePath = toolbox.get_file_path_with_extension_include_subfolders("../../imagePool/NIST/by_write/hsf_0/f0000_14", ["png"])
imagePath = F2.load_image(imagePath[:100], True, True, 64, 64)

if imagePath == []:
    print("no files")
    exit()

# ---------------------------------------------
# F2: Generate and save scatter plate
# ---------------------------------------------
toolbox.print_program_section_name("F2: Generate and save scatter plate")
      
scatterPlateRandom = F2.create_scatter_plate()
#scatterPlateRandom = ['data/F2/intermediateData/scatterPlate/ScatterPlateRandomX', 'data/F2/intermediateData/scatterPlate/ScatterPlateRandomY']

# -------------------------------------------------
# F2: Load scatter plate and calculate speckle
# -------------------------------------------------
toolbox.print_program_section_name("F2: Load scatter plate and calculate speckle")
      

imagePath = toolbox.get_file_path_with_extension("data/F2/input/NIST/", ["bmp"])
        
F2.calculate_propagation(imagePath, scatterPlateRandom)


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

# ---------------------------------------------
# NEURONAL NETWORK: load data
# ---------------------------------------------
neuronalNetwork.generateFolderStructure()
toolbox.print_program_section_name("NEURONAL NETWORK: load data")

print("the model is loading...")
model = neuronalNetwork.loadModel()
print("done")

print("the training and test datasets are loading...")
imagePath = toolbox.get_file_path_with_extension("data/F2/input/NIST", ["bmp"])
imageGroundTruthPath = neuronalNetwork.loadGroundTruthData(imagePath[:-10])
imageTestGroundTruthPath = neuronalNetwork.loadTestGroundTruthData(imagePath[-10:])


imagePath = toolbox.get_file_path_with_extension("data/F2/output/speckle/", ["bmp"])
imageSpecklePath = neuronalNetwork.loadTrainingData(imagePath[:-10])
imageTestSpecklePath = neuronalNetwork.loadTestData(imagePath[-10:])
print("done")




# ---------------------------------------------
# NEURONAL NETWORK: train network
# ---------------------------------------------
toolbox.print_program_section_name("NEURONAL NETWORK: train network")

model = neuronalNetwork.trainNetwork(imageSpecklePath, imageGroundTruthPath, model, fitEpochs=3, fitBatchSize=10)


# ---------------------------------------------
# NEURONAL NETWORK: test network
# ---------------------------------------------
toolbox.print_program_section_name("NEURONAL NETWORK: test network")

neuronalNetwork.testNetwork(imageTestSpecklePath, model)


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
