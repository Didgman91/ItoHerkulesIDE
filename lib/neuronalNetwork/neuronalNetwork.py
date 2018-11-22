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

from ..toolbox import toolbox

from .model import get_model_deep_speckle
from .model import getLossFunction
from .model import getMetric

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
pathInputTestDataGroundTruth = "/neuronalNetwork/input/testData/groundTruth"
pathInputPretrainedWeights = "/neuronalNetwork/input/pretrainedWeights"
pathIntermediateDataTrainedWeights = "/neuronalNetwork/intermediateData/trainedWeights"
pathOutputPredictions = "/neuronalNetwork/output/predictions"
pathOutputDocumentation = "/neuronalNetwork/output/documentation"

fileNameTrainedWeights = "weights.hdf5"
fileNamePredictions = "prediction.npy"

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
    model = get_model_deep_speckle()
    
    return model

def loadTrainingData(imagePath):
    path = toolbox.loadImage(imagePath, pathData + pathInputTrainingData)
    return path

def loadGroundTruthData(imagePath):
    path = toolbox.loadImage(imagePath, pathData + pathInputTrainingDataGroundTruth)
    return path

def loadTestData(imagePath):
    path = toolbox.loadImage(imagePath, pathData + pathInputTestData)
    return path

def loadTestGroundTruthData(imagePath):
    path = toolbox.loadImage(imagePath, pathData + pathInputTestDataGroundTruth)
    return path

def getImageAsNpy(imagePath):
    ""
    rv = []
    for ip in imagePath:
        fileExtension = os.path.splitext(ip)[-1]

#        base = os.path.basename(ip)
#        name = os.path.splitext(base)
        data = []
        
        if fileExtension == ".bmp":
            img = Image.open(ip)#.convert('LA')
            img.load()
            data = np.asarray(img, dtype="int32")/255
        elif (fileExtension == ".npy") | (fileExtension == ".bin"):
            data = np.load(ip)
            data = Image.fromarray(np.uint8(data*255))#.convert('LA')
            data = np.asarray(data, dtype="int32")/255
        else:
            continue
        
        rv = rv + [data]
    
    rv = convertImageListToNpArray4d(rv)
    
    return rv

def convertImageListToNpArray4d(images):
    "4d array: 1dim: image number, 2d: x dimension of the image, 3d: y dimension of the image, 4d: channel"
    if type(images) is list:
        # stack a list to a numpy array
        images = np.stack((images))
    elif type(images) is np.array:
        buffer=0    # do nothing
    else:
        print("error: type of _images_ not supported")
        return 0
    
    # convert RGB image to gray scale image
    if len(images.shape) == 4:
        if images.shape[3] == 3:
            buffer = []
            for i in images:
                b = Image.fromarray(np.uint8(i*255)).convert('L')
                buffer += [np.asarray(b, dtype="int32")]
                
            images = np.stack((buffer))
    if images.max() > 1:
        images = images / 255
    
    # reshape the numpy array (imageNumber, xPixels, yPixels, 1)
    if len(images.shape) == 3:
        images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))        
    
    return images

def save4dNpyAsBmp(npyPath, filename, bmpFolderPath=""):
    if bmpFolderPath == "":
        bmpFolderPath = pathData + pathOutputPredictions
        
    npy = np.load(npyPath)
    
    for i in range(len(npy)):
        image = Image.fromarray(np.uint8(npy[i]*255)).convert('RGB')
        image.save(pathData + pathOutputPredictions + "/{}.bmp".format(filename[i]))

def trainNetwork(trainingDataPath, groundTruthPath, model, fitEpochs, fitBatchSize):
    trainingData = getImageAsNpy(trainingDataPath)
    groundTruth = getImageAsNpy(groundTruthPath)
    
    # Compile model
#    model.compile(loss={'predictions':getLossFunction}, optimizer='adam', metrics=['accuracy', getMetric])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    
    
    # Fit the model
    model.fit(trainingData, groundTruth, epochs=fitEpochs, batch_size=fitBatchSize)
    
    model.save_weights(pathData + pathIntermediateDataTrainedWeights + "/" + fileNameTrainedWeights)
    
    return model

def testNetwork(testDataPath, model, trainedWeightsPath=""):
    testData = getImageAsNpy(testDataPath)
    
    fileName = []
    for ip in testDataPath:
        base = os.path.basename(ip)
        fileName += [os.path.splitext(base)[0]]
    
    if trainedWeightsPath == "":
        model.save_weights(pathData + pathIntermediateDataTrainedWeights + "/" + fileNameTrainedWeights)
    
    pred = model.predict(testData, batch_size=2)
    
    path = pathData + pathOutputPredictions + "/" + fileNamePredictions
    np.save(path, pred)
    
    save4dNpyAsBmp(path, fileName)
    
    return path

