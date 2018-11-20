#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:58:44 2018

@author: maxdh
"""

import os
from shutil import copyfile
import numpy as np

from PIL import Image

from .model import get_model_deep_speckle_64x64

#from __future__ import print_function
#
#from keras.models import Model
#from keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation
#from keras.layers.normalization import BatchNormalization
#from keras.regularizers import l2


pathData = "data"

pathInput = "/neuronalNetwork/input"
pathInputModel = "/neuronalNetwork/input/model"
pathInputTrainingData = "/neuronalNetwork/input/trainingData"
pathInputTrainingDataGroundTruth = "/neuronalNetwork/input/trainingData/groundTruth"
pathInputTestData = "/neuronalNetwork/input/testData"
pathInputPretrainedWeights = "/neuronalNetwork/input/pretrainedWeights"
pathIntermediateDataTrainedWeights = "/neuronalNetwork/intermediateData/trainedWeights"
pathOutputPredictions = "/neuronalNetwork/output/predictions"
pathOutputDocumentation = "/neuronalNetwork/output/documentation"


def generateFolderStructure():
    "Creats folders and subfolders related to the F2 process in the folder _path_."
    os.makedirs(pathData + pathInput, 0o777, True)
    os.makedirs(pathData + pathInputModel, 0o777, True)
    os.makedirs(pathData + pathInputTrainingData, 0o777, True)
    os.makedirs(pathData + pathInputTrainingDataGroundTruth, 0o777, True)
    os.makedirs(pathData + pathInputTestData, 0o777, True)
    os.makedirs(pathData + pathInputPretrainedWeights, 0o777, True)
    os.makedirs(pathData + pathIntermediateDataTrainedWeights, 0o777, True)
    os.makedirs(pathData + pathOutputPredictions, 0o777, True)
    os.makedirs(pathData + pathOutputDocumentation, 0o777, True)

def loadModel():
    modelFile = "lib/neuronalNetwork/model.py"
    copyfile(modelFile, pathData + pathInputModel + "/model.py")
    
    # model is defined in model.py
    model = get_model_deep_speckle_64x64()
    
    return model

def loadBMPasNPY(imagePath):
    rv = []
    for ip in imagePath:
        fileExtension = os.path.splitext(ip)[-1]
        if fileExtension == ".bmp":
            base = os.path.basename(ip)
            name = os.path.splitext(base)
            
            img = Image.open(ip)
            img.load()
            data = np.asarray(img, dtype="int32")
            
            buffer = pathData + pathInputTrainingData + "/{}.npy".format(name[0]) 
            np.save(buffer, data)
            rv = rv + [buffer]
        
    return rv

def trainNetwork(inputData, groundTruth):
    model = loadModel()
    
    
    
    return

def testNetwork():
    
    return

