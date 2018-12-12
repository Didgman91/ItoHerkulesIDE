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
from scipy.stats import pearsonr

from PIL import Image
from PIL import ImageOps

import keras
import keras.backend as K
import tensorflow as tf

from ..toolbox import toolbox

from .losses import pcc
from .losses import jd

#from __future__ import print_function
#
#from keras.models import Model
#from keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation
#from keras.layers.normalization import BatchNormalization
#from keras.regularizers import l2


# def generate_arrays(data, ground_Truth, batch_size):
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
# for i in range(len(data)):
# img = Image.open(ground_Truth[i])#.convert('LA')
# img.load()
##            dataNpy = np.asarray(img, dtype="int32")/255
##
# img = Image.open(data[i])#.convert('LA')
# img.load()
##            ground_TruthNpy = np.asarray(img, dtype="int32")/255
##
# yield(dataNpy, ground_TruthNpy)
#
# def get_Image_As_Npy(image_Path):
#    rv = []
#    for i in range(len(image_Path)):
#            img = Image.open(image_Path[i])#.convert('LA')
#            img.load()
#            rv += [np.asarray(img, dtype="int32")/255]

def convert_Image_List_To_4D_Np_Array(images):
    """Converts an image to an 4d array.

     With the dimensions:
         - 1d: image number
         - 2d: x dimension of the image
         - 3d: y dimension of the image
         - 4d: channel
    
    Arguments
    ----
        images
    
    Returns
    ----
    
    """
    if (isinstance(images, list)) & (len(images) > 1):
        # stack a list to a numpy array
        images = np.stack((images))
    elif isinstance(images, np.array):
        buffer = 0    # do nothing
    else:
        print("error: type of _images_ not supported")
        print("Type: {}".format(type(images)))
        return 0

    # convert RGB image to gray scale image
    if len(images.shape) == 4:
        if images.shape[3] == 3:
            buffer = []
            for i in images:
                b = Image.fromarray(np.uint8(i * 255)).convert('L')
                buffer += [np.asarray(b, dtype="int32")]

            images = np.stack((buffer))
    if images.max() > 1:
        images = images / 255

    # reshape the numpy array (imageNumber, xPixels, yPixels, 1)
    if len(images.shape) == 3:
        images = images.reshape(
            (images.shape[0], images.shape[1], images.shape[2], 1))

    return images


class neuronal_Network_Class:

    def __init__(self, model, neuronal_Network_Path_Extension=""):
        """Creats folders and subfolders related to the F2 process in the
        folder _path_.

        Arguments
        ----
            model
                model of the network

            neuronal_Network_Path_Extension
                When a string is specified, input, intermediateData and output
                folder will be stored in a subfolder with that name. If not,
                these folders will be stored directly in the
                path_Neuronal_Network_Data folder.
        """

        self.model = model

        self.path_Data = "data"
        self.path_Neuronal_Network_Data = "/neuronal_network"

        if neuronal_Network_Path_Extension != "":
            if neuronal_Network_Path_Extension[-1] == '/':
                neuronal_Network_Path_Extension = neuronal_Network_Path_Extension[:-1]
            if neuronal_Network_Path_Extension[0] != '/':
                neuronal_Network_Path_Extension = "/" + neuronal_Network_Path_Extension

        print(
            "path_Neuronal_Network_Data: {}".format(
                self.path_Data +
                self.path_Neuronal_Network_Data))

        self.path_Input = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/input"
        self.path_Input_Model = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/input/model"
        self.path_Input_Training_Data = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/input/training_data"
        self.path_Input_Training_Data_Ground_Truth = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/input/training_data/ground_truth"
        self.path_Input_Validation_Data = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/input/validation_data"
        self.path_Input_Validation_Data_Ground_Truth = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/input/validation_data/ground_truth"
        self.path_Input_Test_Data = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/input/test_data"
        self.path_Input_Test_Data_Ground_Truth = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/input/test_data/ground_truth"
        self.path_Input_Pretrained_Weights = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/input/pretrained_weights"

        self.path_Intermediate_Data_Trained_Weights = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/intermediate_data/trained_weights"
        self.path_Intermediate_Data_History = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/intermediate_data/history"

        self.path_Output_Validation_Data_Prediction = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/output/validation_data_predictions"
        self.path_Output_Test_Data_Prediction = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/output/test_data_predictions"
        self.path_Output_Evaluation = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/output/evaluation"
        self.path_Output_Documentation = self.path_Neuronal_Network_Data + \
            neuronal_Network_Path_Extension + "/output/documentation"

        self.file_Name_Trained_Weights = "weights.hdf5"
        self.file_Name_Predictions = "prediction.npy"
        self.file_Name_History = "history.pkl"

        # create input folders
        os.makedirs(self.path_Data + self.path_Input, 0o777, True)
        os.makedirs(self.path_Data + self.path_Input_Model, 0o777, True)
        os.makedirs(
            self.path_Data +
            self.path_Input_Training_Data,
            0o777,
            True)
        os.makedirs(
            self.path_Data +
            self.path_Input_Training_Data_Ground_Truth,
            0o777,
            True)
        os.makedirs(self.path_Data + self.path_Input_Test_Data, 0o777, True)
        os.makedirs(
            self.path_Data +
            self.path_Input_Pretrained_Weights,
            0o777,
            True)

        # create intermediate data folders
        os.makedirs(
            self.path_Data +
            self.path_Intermediate_Data_Trained_Weights,
            0o777,
            True)
        os.makedirs(
            self.path_Data +
            self.path_Intermediate_Data_History,
            0o777,
            True)

        # create output folders
        os.makedirs(
            self.path_Data +
            self.path_Output_Test_Data_Prediction,
            0o777,
            True)
        os.makedirs(
            self.path_Data +
            self.path_Output_Validation_Data_Prediction,
            0o777,
            True)
        os.makedirs(
            self.path_Data +
            self.path_Output_Documentation,
            0o777,
            True)
        os.makedirs(self.path_Data + self.path_Output_Evaluation, 0o777, True)

    def load_Model(self, modelFilePath="",
                   neuronal_Network_Path_ExtensionPretrainedWeights=""):
        """
        Loads the model and copies the _modelFilePath_ into the input folder.

        # Argumets
            modelFilePath
                relative path to the model in parent _path_Data_ folder

            neuronal_Network_Path_ExtensionPretrainedWeights
                name of the subfolder at _path_Neuronal_Network_Data_ path, which
                will be used instead of _path_Neuronal_Network_Data_

        Returns
    ----
            the model object
        """

        if modelFilePath != "":
            copyfile(
                modelFilePath,
                self.path_Data +
                self.path_Input_Model +
                "/model.py")

        # model is defined in model.py
        self.model = get_model_deep_speckle()

        # load weights of the previous fog layer
        if (neuronal_Network_Path_ExtensionPretrainedWeights != ""):
            pos = self.path_Intermediate_Data_Trained_Weights.find(
                "/intermediateData")
            path = self.path_Data \
                + self.path_Neuronal_Network_Data \
                + "/" + neuronal_Network_Path_ExtensionPretrainedWeights \
                + self.path_Intermediate_Data_Trained_Weights[pos:] \
                + "//" + self.file_Name_Trained_Weights

            self.load_Weights(path)
#            copyfile(path, self.path_Data + self.path_Input_Pretrained_Weights + "/" + neuronal_Network_Path_ExtensionPretrainedWeights + "_" + self.file_Name_Trained_Weights)
#            self.model.load_weights(path)

        return self.model

    def load_Weights(self, path=""):
        """
        Loads the weights out of an hdf5 file. The model must be loaded first.

        Arguments
        ----
            path
                path to the hdf5 file

        Returns
        ----
            the model with the pretrained weights.
        """
        if path == "":
            path = self.path_Data \
                    + self.path_Intermediate_Data_Trained_Weights \
                    + "/" + self.file_Name_Trained_Weights
        else:
            copyfile(
                path,
                self.path_Data
                + self.path_Input_Pretrained_Weights
                + "/" + self.file_Name_Trained_Weights)

        self.model.load_weights(path)

        return self.model

    def load_Training_Data(self, image_Path, prefix=""):
        """
        Loads the training data and copies the listed files under
        _image_Path_ into the input folder.

        Arguments
        ----
            image_Path
                List of relative paths to the training data images in the
                parent folder _path_Data_.

            prefix
                A prefix can be added to the image file during the copy
                operation.

        Returns
        ----
            a list of the copied images
        """
        path = toolbox.load_Image(
            image_Path,
            self.path_Data +
            self.path_Input_Training_Data,
            prefix=prefix)
        return path

    def load_Ground_Truth_Data(self, image_Path,
                               load_Multiple_Times_With_Prefix=[]):
        """Loads the ground truth training data and copies the listed files
        under _image_Path_ into the input folder.

        Arguments
        ----
            image_Path
                List of relative paths to the ground truth training data images
                in the parent folder _path_Data_.

            load_Multiple_Times_With_Prefix
                If this is set, than all images listed under _image_Path_ are
                saved multiple times with the listed prefix.

        Returns
        ----
            a list of the copied images
        """
        path = []
        if load_Multiple_Times_With_Prefix == []:
            path = toolbox.load_Image(
                image_Path,
                self.path_Data +
                self.path_Input_Training_Data_Ground_Truth)
        else:
            for prefix in load_Multiple_Times_With_Prefix:
                path += toolbox.load_Image(
                    image_Path,
                    self.path_Data +
                    self.path_Input_Training_Data_Ground_Truth,
                    prefix="{}_".format(prefix))
        return path

    def load_Test_Data(self, image_Path, prefix=""):
        """Loads the test data and copies the listed files under _image_Path_
        into the input folder.

        Arguments
        ----
            image_Path
                List of relative paths to the test data images in the parent
                folder _path_Data_.

            prefix
                A prefix can be added to the image file during the copy
                operation.

        Returns
        ----
            a list of the copied images
        """
        path = toolbox.load_Image(
            image_Path,
            self.path_Data +
            self.path_Input_Test_Data,
            prefix=prefix)
        return path

    def load_Test_Ground_Truth_Data(
            self, image_Path, load_Multiple_With_Prefix=[]):
        """Loads the ground truth test data and copies the listed files under
        _image_Path_ into the input folder.

        Arguments
        ----
            image_Path
                List of relative paths to the ground truth test data images in
                the parent folder _path_Data_.

            load_Multiple_Times_With_Prefix
                If this is set, than all images listed under _image_Path_ are
                saved multiple times with the listed prefix.

        Returns
        ----
            a list of the copied images
        """
        path = []
        if load_Multiple_With_Prefix == []:
            path = toolbox.load_Image(
                image_Path,
                self.path_Data +
                self.path_Input_Test_Data_Ground_Truth)
        else:
            for prefix in load_Multiple_With_Prefix:
                destination = self.path_Data \
                                + self.path_Input_Test_Data_Ground_Truth
                path += toolbox.load_Image(
                        image_Path,
                        destination,
                        prefix="{}_".format(prefix))
        return path

    def load_Validation_Data(self, image_Path, prefix=""):
        """Loads the validation data and copies the listed files under
        _image_Path_ into the input folder.

        Arguments
        ----
            image_Path
                List of relative paths to the validataion data images in the
                parent folder _path_Data_.

            prefix
                A prefix can be added to the image file during the copy
                operation.

        Returns
        ----
            a list of the copied images
        """
        path = toolbox.load_Image(
            image_Path,
            self.path_Data +
            self.path_Input_Validation_Data,
            prefix=prefix)
        return path

    def load_Validation_Ground_Truth_Data(
            self, image_Path, load_Multiple_With_Prefix=[]):
        """Loads the ground truth test data and copies the listed files under
        _image_Path_ into the input folder.

        Arguments
        ----
            image_Path
                List of relative paths to the ground truth test data images in
                the parent folder _path_Data_.

            load_Multiple_Times_With_Prefix
                If this is set, than all images listed under _image_Path_ are
                saved multiple times with the listed prefix.

        Returns
        ----
            a list of the copied images
        """
        path = []
        if load_Multiple_With_Prefix == []:
            path = toolbox.load_Image(
                image_Path,
                self.path_Data +
                self.path_Input_Validation_Data_Ground_Truth)
        else:
            for prefix in load_Multiple_With_Prefix:
                destination = self.path_Data \
                                + self.path_Input_Validation_Data_Ground_Truth
                path += toolbox.load_Image(
                        image_Path,
                        destination,
                        prefix="{}_".format(prefix))
        return path

    def get_Image_As_Npy(self, image_Path):
        """ Opens the images listed under _image_Path_ and returns them as a
        list of numpy arrays.

        Arguments
        ----
            image_Path
                List of relative paths to the ground truth test data images in
                the parent folder _path_Data_.
        
        Returns
        ----
            
        """
        rv = []
        for ip in image_Path:
            file_Extension = os.path.splitext(ip)[-1]

    #        base = os.path.basename(ip)
    #        name = os.path.splitext(base)
            data = []

            if file_Extension == ".bmp":
                img = Image.open(ip)  # .convert('LA')
                img.load()
                data = np.asarray(img, dtype="int32") / 255
            elif (file_Extension == ".npy") | (file_Extension == ".bin"):
                data = np.load(ip)
                data = Image.fromarray(np.uint8(data * 255))  # .convert('LA')
                data = np.asarray(data, dtype="int32") / 255
            else:
                continue

            rv = rv + [data]

        rv = convert_Image_List_To_4D_Np_Array(rv)

        return rv

    def save_4D_Npy_As_Bmp(self, npy_Array, filename,
                           bmp_Folder_Path, invert_Color=False):
        """Saves the 4d numpy array as bmp files.

        Dimensions:
            - 1d: image number
            - 2d: x dimension of the image
            - 3d: y dimension of the image
            - 4d: channel

        Arguments
        ----
            npy_Array
                numpy array

            filename
                list of bmp filenames

            bmp_Folder_Path
                Destination where the bmp files will be stored.

            invert_Color
                If it's True, than the colors of the images will be inverted.

        Returns
        ----
            the bmp file paths.
        """
        path = []
        for i in range(len(npy_Array)):
            image = toolbox.convert_3d_Npy_To_Image(npy_Array[i], invert_Color)

            buffer = bmp_Folder_Path + "/{}.bmp".format(filename[i])
            image.save(buffer)

            path += [buffer]

        return path

    def train_Network(self, training_Data_Path,
                      ground_Truth_Path, fit_Epochs, fit_Batch_Size):
        """ Runs the routine to train the network.

        Arguments
        ----
            training_Data_Path
                list of paths to images

            ground_Truth_Path
                list of paths to images

            fit_Epochs
                number of epochs

            fit_Batch_Size
                batch size
        """
        training_Data_Path.sort()
        ground_Truth_Path.sort()

        training_Data = self.get_Image_As_Npy(training_Data_Path)
        ground_Truth = self.get_Image_As_Npy(ground_Truth_Path)

        # Compile model
        adam = keras.optimizers.Adam(
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.001 /
            fit_Epochs,
            amsgrad=False)
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=adam)

        # Fit the model
        history = self.model.fit(training_Data,
                                 ground_Truth,
                                 epochs=fit_Epochs,
                                 batch_size=fit_Batch_Size,
                                 verbose=2)
# todo:
#        history = model.fit_generator(generate_arrays(training_Data_Path, ground_Truth_Path, fit_Batch_Size),
#                      steps_per_epoch=fit_Batch_Size, epochs=fit_Epochs)

        # saving the history
        with open(self.path_Data + self.path_Intermediate_Data_History + "/"
                  + self.file_Name_History, "wb") as f:
            pickle.dump(history.history, f)

        self.model.save_weights(
                self.path_Data
                + self.path_Intermediate_Data_Trained_Weights
                + "/" + self.file_Name_Trained_Weights)

    def validate_Network(self, validation_Data_Path, trained_Weights_Path=""):
        """Calculates predictions based on the validation dataset.

        Arguments
        ----
            validation_Data_Path
                list of paths to images

            trained_Weights_Path
                If a path is specified, the model uses these weights instead
                of the weights from the intermediateData folder.

        Returns
        ----
            the path of the prediction.
        """
        path = self.predict(
            validation_Data_Path,
            self.path_Data +
            self.path_Output_Validation_Data_Prediction,
            trained_Weights_Path)

        return path

    def test_Network(self, test_Data_Path, trained_Weights_Path=""):
        """Calculates predictions based on the test dataset.

        Arguments
        ----
            test_Data_Path
                list of paths to images

            trained_Weights_Path
                If a path is specified, the model uses these weights instead
                of the weights from the intermediateData folder.

        Returns
        ----
            the path of the prediction.
        """
        path = self.predict(
            test_Data_Path,
            self.path_Data +
            self.path_Output_Test_Data_Prediction,
            trained_Weights_Path)

        return path

    def predict(self, data_Path, prediction_Folder_Path,
                trained_Weights_Path=""):
        """Calculates pridictions based on the given data_Path.

        Arguments
        ----
            data_Path
                path of the input data

            prediction_Folder_Path
                folder where the prediction shoud be stored

            trained_Weights_Path
                If a path is specified, the model uses these weights instead
                of the weights from the intermediateData folder.

        Returns
        ----
            the path of the prediction.
        """
        data_Path.sort()
        test_Data = self.get_Image_As_Npy(data_Path)

        fileName = []
        for ip in data_Path:
            base = os.path.basename(ip)
            fileName += [os.path.splitext(base)[0]]

        if trained_Weights_Path == "":
            self.model.save_weights(
                self.path_Data +
                self.path_Intermediate_Data_Trained_Weights +
                "/" +
                self.file_Name_Trained_Weights)

        pred = self.model.predict(test_Data)

        path = self.save_4D_Npy_As_Bmp(
            pred, fileName, prediction_Folder_Path, invert_Color=True)

        return path


    def jaccard_index(self, path_ground_truth, path_prediction, parameter = []):
        """Calculates the Jaccard index by comparing the images in both paths
        and stores them in a csv file. Optionally these values can be saved as
        an image (export_images).
        
        Arguments
        ----
            path_Ground_Truth
                list of paths for the ground truth dataset
            
            path_Prediction
                list of paths for the prediction dataset
            
            parameter
                optional list of parameters

        Parameter
        ----
        parameter[0]: export_images
            export the Jaccard index as images [default: True]
        """
        if parameter != []:
            export_images = parameter[0]
        else:
            export_images = True

        path_prediction.sort()
        path_ground_truth.sort()

        pred = self.get_Image_As_Npy(path_prediction)
        ground_truth = self.get_Image_As_Npy(path_ground_truth)

        # calculate jaccard_distance
        sess = tf.InteractiveSession()

        score = jd(ground_truth, pred)
        scoreNP = -1*(score.eval(session=sess)-1)

        meanNP_per_image = []
        for i in range(len(scoreNP[:, 0, 0])):
            meanNP_per_image += [np.mean(scoreNP[i, :, :])]

        mean = K.mean(score)
        meanNP = -1*(mean.eval(session=sess)-1)

        sess.close() 
        
        # save calculations
        file_name = "jaccard_index.txt"
        path_ji= self.path_Output_Evaluation + "/jaccard_index"
        path_ji_images = self.path_Output_Evaluation + "/jaccard_index/images"
        os.makedirs(self.path_Data + path_ji, 0o777, True)
        
        image_file_name = []
        for i in range(len(path_prediction)):
            image_file_name += [toolbox.get_File_Name(path_prediction[i])]
        image_file_name = np.stack(image_file_name)
        
        file = self.path_Data + path_ji + "/" + file_name
        with open(file, 'w') as f:
            f.write("object name;jaccard index\n")
            f.write("mean;{}\n".format(meanNP))
            for i in range(len(meanNP_per_image)):
                f.write("{};{}\n".format(image_file_name[i], meanNP_per_image[i]))

        # ji as images
        if export_images is True:
            os.makedirs(self.path_Data + path_ji_images, 0o777, True)
            for i in range(len(scoreNP[:,0,0])):
                image = toolbox.convert_3d_Npy_To_Image(scoreNP[i,:,:], invert_Color=False)
    
                buffer = self.path_Data + path_ji_images + "/{}.bmp".format(image_file_name[i])
                image.save(buffer)
    
    def get_pcc_base(self, np_array_1, np_array_2):
        """Calculates the Pearson correlation coefficient of two numpy arrays.
        
        Arguments
        ----
            np_array_1
                first array compared to np_array_2
                
            np_array_2
                must be the same size as the np_array_1
                
        Returns
        ----
            the Pearson correlation coefficient.
        """
        gt1d = np_array_1.flatten()
        pred1d = np_array_2.flatten()
        
        value, p = pearsonr(gt1d, pred1d)
        
        return value
        
    
    def pearson_correlation_coefficient(self, path_ground_truth, path_prediction, parameter = []):
        """Calculates the Pearson correlation coefficient by comparing the
        images in both paths and stores them in a csv file.
        
        Arguments
        ----
            path_Ground_Truth
                list of paths for the ground truth dataset
            
            path_Prediction
                list of paths for the prediction dataset
            
            parameter
                optional list of parameters

        Parameter
        ----
        parameter[0]: use_tensorflow
            use the tensorflow library instead of scipy [default: False]
        """
        if parameter != []:
            use_tensorflow = parameter[0]
        else:
            use_tensorflow = False
        
        path_prediction.sort()
        path_ground_truth.sort()

        pred = self.get_Image_As_Npy(path_prediction)
        ground_truth = self.get_Image_As_Npy(path_ground_truth)
        
        # calculate Pearson correlation coefficient
        value_pcc = []
        mean_pcc = []
        if use_tensorflow is True:
            sess = tf.InteractiveSession()

            for i in range(len(ground_truth)):
                value = pcc(ground_truth[i], pred[i])
                value_pcc += [value.eval(session=sess)]
            
            mean = pcc(ground_truth, pred)
            mean_pcc = mean.eval(session=sess)
            
            sess.close()
        else:
            for i in range(len(ground_truth)):
               value_pcc += [self.get_pcc_base(ground_truth[i], pred[i])]

            mean_pcc = np.mean(value_pcc)
        
        # save calculations
        file_name = "pearson_correlation_coefficient.txt"
        path_pcc= self.path_Output_Evaluation + "/pearson_correlation_coefficient"
        os.makedirs(self.path_Data + path_pcc, 0o777, True)

        image_file_name = []
        for i in range(len(path_prediction)):
            image_file_name += [toolbox.get_File_Name(path_prediction[i])]
        image_file_name = np.stack(image_file_name)
        
        file = self.path_Data + path_pcc + "/" + file_name
        with open(file, 'w') as f:
            f.write("object name;pearson correlation coefficient\n")
            f.write("mean;{}\n".format(mean_pcc))
            for i in range(len(value_pcc)):
                f.write("{};{}\n".format(image_file_name[i], value_pcc[i]))

            
    
    def evaluate_Network(self, method, path_Ground_Truth, path_Prediction,
                         additional_method_parameter = []):
        """Wrapper function to call multiple evaluation methods
        
        Arguments
        ----
            method
                list of functions, but at least it can be a singel function

            path_Ground_Truth
                list of paths for the ground truth dataset
            
            path_Prediction
                list of paths for the prediction dataset
        
        Returns 
        ----
            list of method returns, if any.
        """
        path_Ground_Truth.sort()
        path_Prediction.sort()

        rv = []

        if type(method) is list:
            for i in range(len(method)):
                if additional_method_parameter != []:
                    rv += [method[i](path_Ground_Truth, path_Prediction, additional_method_parameter[i])]
                else:
                    rv += [method[i](path_Ground_Truth, path_Prediction)]
        else:
            if additional_method_parameter != []:
                rv = [method(path_Ground_Truth, path_Prediction, additional_method_parameter)]
            else:
                rv = [method(path_Ground_Truth, path_Prediction)]


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
