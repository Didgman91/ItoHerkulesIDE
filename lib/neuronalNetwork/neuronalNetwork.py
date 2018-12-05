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
#            data_Npy = get_Image_As_Npy(data[batch_start:limit])
#            ground_Truth_Npy = get_Image_As_Npy(ground_Truth[batch_start:limit])
#            
#            data_Npy = convert_Image_List_To_4D_Np_Array(data_Npy)
#            ground_Truth_Npy = convert_Image_List_To_4D_Np_Array(ground_Truth_Npy)
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
##            ground_TruthNpy = np.asarray(img, dtype="int32")/255
##            
##            yield(dataNpy, ground_TruthNpy)
#
#def get_Image_As_Npy(image_Path):
#    rv = []
#    for i in range(len(image_Path)):
#            img = Image.open(image_Path[i])#.convert('LA')
#            img.load()
#            rv += [np.asarray(img, dtype="int32")/255]

def convert_Image_List_To_4D_Np_Array(images):
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

class neuronal_Network_Class:
    
    def __init__(self, neuronal_Network_Path_Extension=""):
        "Creats folders and subfolders related to the F2 process in the folder _path_."
        
        self.path_Data = "data"
        self.path_Neuronal_Network_Data = "/neuronalNetwork"
        
        if neuronal_Network_Path_Extension != "":
            if neuronal_Network_Path_Extension[-1] == '/':
                neuronal_Network_Path_Extension = neuronal_Network_Path_Extension[:-1]
            if neuronal_Network_Path_Extension[0] != '/':
                neuronal_Network_Path_Extension = "/"+ neuronal_Network_Path_Extension
        
        print("path_Neuronal_Network_Data: {}".format(self.path_Data + self.path_Neuronal_Network_Data))
        
        
        self.path_Input = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/input"
        self.path_Input_Model = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/input/model"
        self.path_Input_Training_Data = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/input/training_Data"
        self.path_Input_Training_Dataground_Truth = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/input/trainingData/groundTruth"
        self.path_Input_Test_Data = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/input/test_Data"
        self.path_Input_Test_Data_Ground_Truth = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/input/testData/groundTruth"
        self.path_Input_Pretrained_Weights = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/input/pretrainedWeights"
        
        self.path_Intermediate_Data_Trained_Weights = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/intermediateData/trainedWeights"
        self.path_Intermediate_Data_History = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/intermediateData/history"
        
        self.path_Output_Test_Data_Prediction = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/output/predictions"
        self.path_Output_Evaluation = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/output/evaluation"
        self.path_Output_Documentation = self.path_Neuronal_Network_Data + neuronal_Network_Path_Extension + "/output/documentation"
        
        self.file_Name_Trained_Weights = "weights.hdf5"
        self.file_Name_Predictions = "prediction.npy"
        self.file_Name_History = "history.pkl"
        
        # create input folders
        os.makedirs(self.path_Data + self.path_Input, 0o777, True)
        os.makedirs(self.path_Data + self.path_Input_Model, 0o777, True)
        os.makedirs(self.path_Data + self.path_Input_Training_Data, 0o777, True)
        os.makedirs(self.path_Data + self.path_Input_Training_Dataground_Truth, 0o777, True)
        os.makedirs(self.path_Data + self.path_Input_Test_Data, 0o777, True)
        os.makedirs(self.path_Data + self.path_Input_Pretrained_Weights, 0o777, True)
        
        # create intermediate data folders
        os.makedirs(self.path_Data + self.path_Intermediate_Data_Trained_Weights, 0o777, True)
        os.makedirs(self.path_Data + self.path_Intermediate_Data_History, 0o777, True)
        
        # create output folders
        os.makedirs(self.path_Data + self.path_Output_Test_Data_Prediction, 0o777, True)
        os.makedirs(self.path_Data + self.path_Output_Documentation, 0o777, True)
        os.makedirs(self.path_Data + self.path_Output_Evaluation, 0o777, True)
    
    def load_Model(self, modelFilePath = "", neuronal_Network_Path_ExtensionPretrainedWeights = ""):
        """
        Loads the model and copies the _modelFilePath_ into the input folder.
        
        # Argumets
            modelFilePath
                relative path to the model in parent _path_Data_ folder
            
            neuronal_Network_Path_ExtensionPretrainedWeights
                name of the subfolder at _path_Neuronal_Network_Data_ path, which 
                will be used instead of _path_Neuronal_Network_Data_
            
        # Returns
            the model object
        """
        
        if modelFilePath != "":
            copyfile(modelFilePath, self.path_Data + self.path_Input_Model + "/model.py")
        
        # model is defined in model.py
        self.model = get_model_deep_speckle()
        
        # load weights of the previous fog layer
        if (neuronal_Network_Path_ExtensionPretrainedWeights!=""):
            pos  = self.path_Intermediate_Data_Trained_Weights.find("/intermediateData")
            path = self.path_Data + self.path_Neuronal_Network_Data + "/"+ neuronal_Network_Path_ExtensionPretrainedWeights + self.path_Intermediate_Data_Trained_Weights[pos:] + "//" + self.file_Name_Trained_Weights
            self.load_Weights(path)
#            copyfile(path, self.path_Data + self.path_Input_Pretrained_Weights + "/" + neuronal_Network_Path_ExtensionPretrainedWeights + "_" + self.file_Name_Trained_Weights)
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
            path = self.path_Data + self.path_Intermediate_Data_Trained_Weights + "/" + self.file_Name_Trained_Weights
        else:
            copyfile(path, self.path_Data + self.path_Input_Pretrained_Weights + "/" + self.file_Name_Trained_Weights)
        
        self.model.load_weights(path)
        
        return self.model
    
    def load_Training_Data(self, image_Path, prefix=""):
        """
        Loads the training data and copies the listed files under 
        _image_Path_ into the input folder.
        
        # Arguments
            image_Path
                List of relative paths to the training data images in the parent folder _path_Data_.
                
            prefix
                A prefix can be added to the image file during the copy operation.
                
        # Returns
            a list of the copied images
        """
        path = toolbox.load_Image(image_Path, self.path_Data + self.path_Input_Training_Data, prefix=prefix)
        return path
    
    def load_Ground_Truth_Data(self, image_Path, load_Multiple_Times_With_Prefix = []):
        """
        Loads the ground truth training data and copies the listed files under
        _image_Path_ into the input folder.
        
        # Arguments
            image_Path
                List of relative paths to the ground truth training data images
                in the parent folder _path_Data_.
                
            load_Multiple_Times_With_Prefix
                If this is set, than all images listed under _image_Path_ are
                saved multiple times with the listed prefix.

        # Returns
            a list of the copied images
        """
        path = []
        if load_Multiple_Times_With_Prefix == []:
            path = toolbox.load_Image(image_Path, self.path_Data + self.path_Input_Training_Dataground_Truth)
        else:
            for prefix in load_Multiple_Times_With_Prefix:
                path += toolbox.load_Image(image_Path, self.path_Data + self.path_Input_Training_Dataground_Truth, prefix="{}_".format(prefix))
        return path
    
    def load_Test_Data(self, image_Path, prefix=""):
        """
        Loads the test data and copies the listed files under _image_Path_
        into the input folder.
        
        # Arguments
            image_Path
                List of relative paths to the test data images in the parent
                folder _path_Data_.
                
            prefix
                A prefix can be added to the image file during the copy
                operation.
                
        # Returns
            a list of the copied images
        """
        path = toolbox.load_Image(image_Path, self.path_Data + self.path_Input_Test_Data, prefix=prefix)
        return path
    
    def load_Test_Ground_Truth_Data(self, image_Path, load_Multiple_With_Prefix = []):
        """
        Loads the ground truth test data and copies the listed files under
        _image_Path_ into the input folder.
        
        # Arguments
            image_Path
                List of relative paths to the ground truth test data images in
                the parent folder _path_Data_.
                
            load_Multiple_Times_With_Prefix
                If this is set, than all images listed under _image_Path_ are
                saved multiple times with the listed prefix.

        # Returns
            a list of the copied images
        """
        path = []
        if load_Multiple_With_Prefix == []:
            path = toolbox.load_Image(image_Path, self.path_Data + self.path_Input_Test_Data_Ground_Truth)
        else:
            for prefix in load_Multiple_With_Prefix:
                path += toolbox.load_Image(image_Path, self.path_Data + self.path_Input_Test_Data_Ground_Truth, prefix="{}_".format(prefix))
        return path
    
    def get_Image_As_Npy(self, image_Path):
        """
        Opens the images listed under _image_Path_ and returns them as a list
        of numpy arrays.
        
        #Arguments
            image_Path
                List of relative paths to the ground truth test data images in
                the parent folder _path_Data_.
        """
        rv = []
        for ip in image_Path:
            file_Extension = os.path.splitext(ip)[-1]
    
    #        base = os.path.basename(ip)
    #        name = os.path.splitext(base)
            data = []
            
            if file_Extension == ".bmp":
                img = Image.open(ip)#.convert('LA')
                img.load()
                data = np.asarray(img, dtype="int32")/255
            elif (file_Extension == ".npy") | (file_Extension == ".bin"):
                data = np.load(ip)
                data = Image.fromarray(np.uint8(data*255))#.convert('LA')
                data = np.asarray(data, dtype="int32")/255
            else:
                continue
            
            rv = rv + [data]
        
        rv = convert_Image_List_To_4D_Np_Array(rv)
        
        return rv
    
    def save_4D_Npy_As_Bmp(self, npyPath, filename, bmp_Folder_Path="", invert_Color=False):
        """
        # Arguments
            npyPath
        """
        if bmp_Folder_Path == "":
            bmp_Folder_Path = self.path_Data + self.path_Output_Test_Data_Prediction
            
        npy = np.load(npyPath)
        
        for i in range(len(npy)):
            image = toolbox.convert_3d_Npy_To_Image(npy[i], invert_Color)
            image.save(self.path_Data + self.path_Output_Test_Data_Prediction + "/{}.bmp".format(filename[i]))
    
    
                
    
    def train_Network(self, training_Data_Path, ground_Truth_Path, model, fit_Epochs, fit_Batch_Size):
        
        training_Data_Path.sort()
        ground_Truth_Path.sort()
        
        training_Data = self.get_Image_As_Npy(training_Data_Path)
        ground_Truth = self.get_Image_As_Npy(ground_Truth_Path)
        
        # Compile model
    #    model.compile(loss={'predictions':getLossFunction}, optimizer='adam', metrics=['accuracy', getMetric])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        
        
        # Fit the model
        history = model.fit(training_Data, ground_Truth, epochs=fit_Epochs, batch_size=fit_Batch_Size, verbose=2)
#        history = model.fit_generator(generate_arrays(training_Data_Path, ground_Truth_Path, fit_Batch_Size),
#                                      steps_per_epoch=fit_Batch_Size, epochs=fit_Epochs)
        
        # saving the history
        with open (self.path_Data + self.path_Intermediate_Data_History + "/" + self.file_Name_History, "wb") as f:
            pickle.dump(history.history, f)
        
        
        model.save_weights(self.path_Data + self.path_Intermediate_Data_Trained_Weights + "/" + self.file_Name_Trained_Weights)
        
        return model
    
    def validate_Network(self, validation_Data_Path, model, trained_Weights_Path=""): 
        validation_Data_Path.sort()
    
    def test_Network(self, test_Data_Path, model, trained_Weights_Path=""):
        test_Data_Path.sort()
        test_Data = self.get_Image_As_Npy(test_Data_Path)
        
        fileName = []
        for ip in test_Data_Path:
            base = os.path.basename(ip)
            fileName += [os.path.splitext(base)[0]]
        
        if trained_Weights_Path == "":
            model.save_weights(self.path_Data + self.path_Intermediate_Data_Trained_Weights + "/" + self.file_Name_Trained_Weights)
        
        pred = model.predict(test_Data)
        
        # todo: don't save save the prediction as a numpy array
        path = self.path_Data + self.path_Output_Test_Data_Prediction + "/" + self.file_Name_Predictions
        np.save(path, pred)
        
        self.save_4D_Npy_As_Bmp(path, fileName, invert_Color=True)
        
        return path

    def evaluate_Network(self, method, path_Ground_Truth=[], path_Prediction=[]):
        if path_Ground_Truth==[]:
            path_Ground_Truth = self.path_Data + self.path_Input_Test_Data_Ground_Truth
            path_Ground_Truth = toolbox.get_file_path_with_extension(path_Ground_Truth, ["bmp"])
                        
        if path_Prediction==[]:
            path_Prediction = self.path_Data + self.path_Output_Test_Data_Prediction
            path_Prediction = toolbox.get_file_path_with_extension(path_Prediction, ["bmp"])
            
        path_Prediction.sort()
        path_Ground_Truth.sort()
        
        pred = self.get_Image_As_Npy(path_Prediction)
        ground_Truth = self.get_Image_As_Npy(path_Ground_Truth)
        
        # calculate jaccard_distance
        sess = tf.InteractiveSession()
            
        score = KC.losses.jaccard_distance(ground_Truth, pred)
        scoreNP = score.eval(session=sess)
        
        meanNP_per_image = []
        for i in range(len(scoreNP[:,0,0])):
            meanNP_per_image += [np.mean(scoreNP[i,:,:])]
        
        mean = K.mean(score)
        meanNP = mean.eval(session=sess)
        
        
        
#        for i in range(len(scoreNP[:,0,0])):
        
#        pred_tf = tf.convert_to_tensor(pred)
#        ground_Truth_tf = tf.convert_to_tensor(ground_Truth)
        
#        rv = []
#        
#        rv += [pcc(ground_Truth_tf, pred_tf)]
        
#        for m in method:
#            sess = tf.InteractiveSession()
#            
#            score = m(ground_Truth, pred)
#            mean = K.mean(score)
#            rv += [mean.eval()]
            
#        return rv

    def loadHistory():
        with open('history.pkl', 'rb') as handle:
            hist = pickle.load(handle)