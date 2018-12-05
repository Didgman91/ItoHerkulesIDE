#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:58:44 2018

@author: maxdh
"""

import os
from shutil import copyfile

import pickle

import numpy as np

from PIL import Image
from PIL import ImageOps

import keras.backend as K
import keras_contrib as KC
import tensorflow as tf

from ..toolbox import toolbox

from .model import get_model_deep_speckle
from .losses import pcc
from .losses import jd


#from __future__ import print_function
#
#from keras.models import Model
#from keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation
#from keras.layers.normalization import BatchNormalization
#from keras.regularizers import l2


#def generate_arrays(data, ground_Truth, batch_size):
#    L = len(data)
#    
#    while True:
#        
#        batch_start = 0
#        batch_end = batch_size
#        
#        while batch_start < L:
#            limit = min(batch_end, L)
#            
#            data_Npy = getImageAsNpy(data[batch_start:limit])
#            ground_Truth_Npy = getImageAsNpy(ground_Truth[batch_start:limit])
#            
#            data_Npy = convertImageListToNpArray4d(data_Npy)
#            ground_Truth_Npy = convertImageListToNpArray4d(ground_Truth_Npy)
#            
#            yield(data_Npy, ground_Truth_Npy)
#            
#            batch_start += batch_size
#            batch_end += batch_size
#            
##        for i in range(len(data)):
##            img = Image.open(ground_Truth[i])#.convert('LA')
##            img.load()
##            dataNpy = np.asarray(img, dtype="int32")/255
##            
##            img = Image.open(data[i])#.convert('LA')
##            img.load()
##            groundTruthNpy = np.asarray(img, dtype="int32")/255
##            
##            yield(dataNpy, groundTruthNpy)
#
#def getImageAsNpy(imagePath):
#    rv = []
#    for i in range(len(imagePath)):
#            img = Image.open(imagePath[i])#.convert('LA')
#            img.load()
#            rv += [np.asarray(img, dtype="int32")/255]

def convertImageListToNpArray4d(images):
        "4d array: 1dim: image number, 2d: x dimension of the image, 3d: y dimension of the image, 4d: channel"
        if (type(images) is list) & (len(images)>1):
            # stack a list to a numpy array
            images = np.stack((images))
        elif type(images) is np.array:
            buffer=0    # do nothing
        else:
            print("error: type of _images_ not supported")
            print("Type: {}".format(type(images)))
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

class neuronalNetworkClass:
    
    def __init__(self, neuronalNetworkPathExtension=""):
        "Creats folders and subfolders related to the F2 process in the folder _path_."
        
        self.pathData = "data"
        self.pathNeuronalNetworkData = "/neuronalNetwork"
        
        if neuronalNetworkPathExtension != "":
            if neuronalNetworkPathExtension[-1] == '/':
                neuronalNetworkPathExtension = neuronalNetworkPathExtension[:-1]
            if neuronalNetworkPathExtension[0] != '/':
                neuronalNetworkPathExtension = "/"+ neuronalNetworkPathExtension
        
        print("pathNeuronalNetworkData: {}".format(self.pathData + self.pathNeuronalNetworkData))
        
        
        self.pathInput = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/input"
        self.pathInputModel = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/input/model"
        self.pathInputTrainingData = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/input/trainingData"
        self.pathInputTrainingDataGroundTruth = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/input/trainingData/groundTruth"
        self.pathInputTestData = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/input/testData"
        self.pathInputTestDataGroundTruth = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/input/testData/groundTruth"
        self.pathInputPretrainedWeights = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/input/pretrainedWeights"
        
        self.pathIntermediateDataTrainedWeights = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/intermediateData/trainedWeights"
        self.pathIntermediateDataHistory = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/intermediateData/history"
        
        self.pathOutputTestDataPrediction = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/output/predictions"
        self.pathOutputEvaluation = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/output/evaluation"
        self.pathOutputDocumentation = self.pathNeuronalNetworkData + neuronalNetworkPathExtension + "/output/documentation"
        
        self.fileNameTrainedWeights = "weights.hdf5"
        self.fileNamePredictions = "prediction.npy"
        self.fileNmaeHistory = "history.pkl"
        
        # create input folders
        os.makedirs(self.pathData + self.pathInput, 0o777, True)
        os.makedirs(self.pathData + self.pathInputModel, 0o777, True)
        os.makedirs(self.pathData + self.pathInputTrainingData, 0o777, True)
        os.makedirs(self.pathData + self.pathInputTrainingDataGroundTruth, 0o777, True)
        os.makedirs(self.pathData + self.pathInputTestData, 0o777, True)
        os.makedirs(self.pathData + self.pathInputPretrainedWeights, 0o777, True)
        
        # create intermediate data folders
        os.makedirs(self.pathData + self.pathIntermediateDataTrainedWeights, 0o777, True)
        os.makedirs(self.pathData + self.pathIntermediateDataHistory, 0o777, True)
        
        # create output folders
        os.makedirs(self.pathData + self.pathOutputTestDataPrediction, 0o777, True)
        os.makedirs(self.pathData + self.pathOutputDocumentation, 0o777, True)
        os.makedirs(self.pathData + self.pathOutputEvaluation, 0o777, True)
    
    def loadModel(self, modelFilePath = "", neuronalNetworkPathExtensionPretrainedWeights = ""):
        """
        Loads the model and copies the _modelFilePath_ into the input folder.
        
        # Argumets
            modelFilePath
                relative path to the model in parent _pathData_ folder
            
            neuronalNetworkPathExtensionPretrainedWeights
                name of the subfolder at _pathNeuronalNetworkData_ path, which 
                will be used instead of _pathNeuronalNetworkData_
            
        # Returns
            the model object
        """
        
        if modelFilePath != "":
            copyfile(modelFilePath, self.pathData + self.pathInputModel + "/model.py")
        
        # model is defined in model.py
        self.model = get_model_deep_speckle()
        
        # load weights of the previous fog layer
        if (neuronalNetworkPathExtensionPretrainedWeights!=""):
            pos  = self.pathIntermediateDataTrainedWeights.find("/intermediateData")
            path = self.pathData + self.pathNeuronalNetworkData + "/"+ neuronalNetworkPathExtensionPretrainedWeights + self.pathIntermediateDataTrainedWeights[pos:] + "//" + self.fileNameTrainedWeights
            self.load_Weights(path)
#            copyfile(path, self.pathData + self.pathInputPretrainedWeights + "/" + neuronalNetworkPathExtensionPretrainedWeights + "_" + self.fileNameTrainedWeights)
#            self.model.load_weights(path)
        
        return self.model
    
    def load_Weights(self, path=""):
        """
        Loads the weights out of an hdf5 file. The model must be loaded first.
        
        # Arguments
            path
                path to the hdf5 file
                
        # Returns
            the model with the pretrained weights.
        """
        if path=="":
            path = self.pathData + self.pathIntermediateDataTrainedWeights + "/" + self.fileNameTrainedWeights
        else:
            copyfile(path, self.pathData + self.pathInputPretrainedWeights + "/" + self.fileNameTrainedWeights)
        
        self.model.load_weights(path)
        
        return self.model
    
    def loadTrainingData(self, imagePath, prefix=""):
        """
        Loads the training data and copies the listed files under 
        _imagePath_ into the input folder.
        
        # Arguments
            imagePath
                List of relative paths to the training data images in the parent folder _pathData_.
                
            prefix
                A prefix can be added to the image file during the copy operation.
                
        # Returns
            a list of the copied images
        """
        path = toolbox.loadImage(imagePath, self.pathData + self.pathInputTrainingData, prefix=prefix)
        return path
    
    def loadGroundTruthData(self, imagePath, loadMultipleTimesWithPrefix = []):
        """
        Loads the ground truth training data and copies the listed files under
        _imagePath_ into the input folder.
        
        # Arguments
            imagePath
                List of relative paths to the ground truth training data images
                in the parent folder _pathData_.
                
            loadMultipleTimesWithPrefix
                If this is set, than all images listed under _imagePath_ are
                saved multiple times with the listed prefix.

        # Returns
            a list of the copied images
        """
        path = []
        if loadMultipleTimesWithPrefix == []:
            path = toolbox.loadImage(imagePath, self.pathData + self.pathInputTrainingDataGroundTruth)
        else:
            for prefix in loadMultipleTimesWithPrefix:
                path += toolbox.loadImage(imagePath, self.pathData + self.pathInputTrainingDataGroundTruth, prefix="{}_".format(prefix))
        return path
    
    def loadTestData(self, imagePath, prefix=""):
        """
        Loads the test data and copies the listed files under _imagePath_
        into the input folder.
        
        # Arguments
            imagePath
                List of relative paths to the test data images in the parent
                folder _pathData_.
                
            prefix
                A prefix can be added to the image file during the copy
                operation.
                
        # Returns
            a list of the copied images
        """
        path = toolbox.loadImage(imagePath, self.pathData + self.pathInputTestData, prefix=prefix)
        return path
    
    def loadTestGroundTruthData(self, imagePath, loadMultipleWithPrefix = []):
        """
        Loads the ground truth test data and copies the listed files under
        _imagePath_ into the input folder.
        
        # Arguments
            imagePath
                List of relative paths to the ground truth test data images in
                the parent folder _pathData_.
                
            loadMultipleTimesWithPrefix
                If this is set, than all images listed under _imagePath_ are
                saved multiple times with the listed prefix.

        # Returns
            a list of the copied images
        """
        path = []
        if loadMultipleWithPrefix == []:
            path = toolbox.loadImage(imagePath, self.pathData + self.pathInputTestDataGroundTruth)
        else:
            for prefix in loadMultipleWithPrefix:
                path += toolbox.loadImage(imagePath, self.pathData + self.pathInputTestDataGroundTruth, prefix="{}_".format(prefix))
        return path
    
    def getImageAsNpy(self, imagePath):
        """
        Opens the images listed under _imagePath_ and returns them as a list
        of numpy arrays.
        
        #Arguments
            imagePath
                List of relative paths to the ground truth test data images in
                the parent folder _pathData_.
        """
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
    
    def convert3dNpyToImage(npy, invert_Color=False):
        image = Image.fromarray(np.uint8(npy*255)).convert('RGB')
        if (invert_Color == True):
            image = ImageOps.invert(image)
                
        return image
    
    def save4dNpyAsBmp(self, npyPath, filename, bmpFolderPath="", invert_Color=False):
        """
        # Arguments
            npyPath
        """
        if bmpFolderPath == "":
            bmpFolderPath = self.pathData + self.pathOutputTestDataPrediction
            
        npy = np.load(npyPath)
        
        for i in range(len(npy)):
            image = convert3dNpyToImage(npy[i], invert_Color)
            image.save(self.pathData + self.pathOutputTestDataPrediction + "/{}.bmp".format(filename[i]))
    
    
                
    
    def trainNetwork(self, trainingDataPath, groundTruthPath, model, fitEpochs, fitBatchSize):
        
        trainingDataPath.sort()
        groundTruthPath.sort()
        
        trainingData = self.getImageAsNpy(trainingDataPath)
        groundTruth = self.getImageAsNpy(groundTruthPath)
        
        # Compile model
    #    model.compile(loss={'predictions':getLossFunction}, optimizer='adam', metrics=['accuracy', getMetric])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        
        
        # Fit the model
        history = model.fit(trainingData, groundTruth, epochs=fitEpochs, batch_size=fitBatchSize)
#        history = model.fit_generator(generate_arrays(trainingDataPath, groundTruthPath, fitBatchSize),
#                                      steps_per_epoch=fitBatchSize, epochs=fitEpochs)
        
        # saving the history
        with open (self.pathData + self.pathINtermediateDataHistory + "/" + self.fileNmaeHistory, "wb") as f:
            pickle.dump(history.history, f)
        
        
        model.save_weights(self.pathData + self.pathIntermediateDataTrainedWeights + "/" + self.fileNameTrainedWeights)
        
        return model
    
    def validate_Network(self, validationDataPath, model, trainedWeightsPath=""):
        validationDataPath.sort()
    
    def testNetwork(self, testDataPath, model, trainedWeightsPath=""):
        testDataPath.sort()
        testData = self.getImageAsNpy(testDataPath)
        
        fileName = []
        for ip in testDataPath:
            base = os.path.basename(ip)
            fileName += [os.path.splitext(base)[0]]
        
        if trainedWeightsPath == "":
            model.save_weights(self.pathData + self.pathIntermediateDataTrainedWeights + "/" + self.fileNameTrainedWeights)
        
        pred = model.predict(testData)
        
        # todo: don't save save the prediction as a numpy array
        path = self.pathData + self.pathOutputTestDataPrediction + "/" + self.fileNamePredictions
        np.save(path, pred)
        
        self.save4dNpyAsBmp(path, fileName, invert_Color=True)
        
        return path

    def evaluate_Network(self, method, path_Ground_Truth=[], path_Prediction=[]):
        if path_Ground_Truth==[]:
            path_Ground_Truth = self.pathData + self.pathInputTestDataGroundTruth
            path_Ground_Truth = toolbox.get_file_path_with_extension(path_Ground_Truth, ["bmp"])
                        
        if path_Prediction==[]:
            path_Prediction = self.pathData + self.pathOutputTestDataPrediction
            path_Prediction = toolbox.get_file_path_with_extension(path_Prediction, ["bmp"])
            
        path_Prediction.sort()
        path_Ground_Truth.sort()
        
        pred = self.getImageAsNpy(path_Prediction)
        groundTruth = self.getImageAsNpy(path_Ground_Truth)
        
        # calculate jaccard_distance
        sess = tf.InteractiveSession()
            
        score = KC.losses.jaccard_distance(groundTruth, pred)
        scoreNP = score.eval(session=sess)
        
        meanNP_per_image = []
        for i in range(len(scoreNP[:,0,0])):
            meanNP_per_image += [np.mean(scoreNP[i,:,:])]
        
        mean = K.mean(score)
        meanNP = mean.eval(session=sess)
        
        self.save4dNpyAsBmp
        
#        for i in range(len(scoreNP[:,0,0])):
        
#        pred_tf = tf.convert_to_tensor(pred)
#        groundTruth_tf = tf.convert_to_tensor(groundTruth)
        
#        rv = []
#        
#        rv += [pcc(groundTruth_tf, pred_tf)]
        
#        for m in method:
#            sess = tf.InteractiveSession()
#            
#            score = m(groundTruth, pred)
#            mean = K.mean(score)
#            rv += [mean.eval()]
            
#        return rv

    def loadHistory():
        with open('history.pkl', 'rb') as handle:
            hist = pickle.load(handle)