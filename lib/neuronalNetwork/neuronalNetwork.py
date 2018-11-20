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
from .model import getLossFunction
from .model import getMetrice

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

def loadTrainingDataAsNpy(imagePath, extension=["npy", "bin", "bmp"]):
    path = loadDataAsNpy(imagePath, pathData + pathInputTrainingData, extension)
    return path

def loadGroundTruthDataAsNpy(imagePath, extension=["npy", "bin", "bmp"]):
    path = loadDataAsNpy(imagePath, pathData + pathInputTrainingDataGroundTruth, extension)
    return path
    
def loadDataAsNpy(imagePath, npyPath, extension=["npy", "bin", "bmp"]):
    "This function will copy all listed files in _imagePath_ and writes a copy in the neuronal network input folder"
    rv = []
    for ip in imagePath:
        fileExtension = os.path.splitext(ip)[-1]
        for ex in extension:
            if ex[0] != ".":
                ex = "." + ex
            
            base = os.path.basename(ip)
            name = os.path.splitext(base)
            data = []
            
            if fileExtension == ".bmp":
                img = Image.open(ip)
                img.load()
                data = np.asarray(img, dtype="int32")
            elif (fileExtension == ".npy") | (fileExtension == ".bin"):
                data = np.load(ip)
            else:
                continue
            
            buffer = npyPath + "/{}.npy".format(name[0]) 
            np.save(buffer, data)
            rv = rv + [buffer]
    return rv

def trainNetwork(trainingData, groundTruth, model):
    
    # Compile model
    model.compile(loss={'predictions':getLossFunction}, optimizer='adam', metrics=['accuracy', getMetrice])
    
    
    # Fit the model
    model.fit(trainingData, groundTruth, epochs=1, batch_size=10)
    
    
    return

def testNetwork():
    
    return

