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


# def generate_arrays(data, ground_truth, batch_size):
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
#            data_npy = get_image_as_npy(data[batch_start:limit])
#            ground_truth_npy = get_image_as_npy(ground_truth[batch_start:limit])
#
#            data_npy = convert_image_list_to_4D_np_array(data_npy)
#            ground_truth_npy = convert_image_list_to_4D_np_array(ground_truth_npy)
#
#            yield(data_npy, ground_truth_npy)
#
#            batch_start += batch_size
#            batch_end += batch_size
#
# for i in range(len(data)):
# img = Image.open(ground_truth[i])#.convert('LA')
# img.load()
##            dataNpy = np.asarray(img, dtype="int32")/255
##
# img = Image.open(data[i])#.convert('LA')
# img.load()
##            ground_truthNpy = np.asarray(img, dtype="int32")/255
##
# yield(dataNpy, ground_truthNpy)
#
# def get_image_as_npy(image_path):
#    rv = []
#    for i in range(len(image_path)):
#            img = Image.open(image_path[i])#.convert('LA')
#            img.load()
#            rv += [np.asarray(img, dtype="int32")/255]

def convert_image_list_to_4D_np_array(images):
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


class neuronal_network_class:

    def __init__(self, model, neuronal_network_path_extension=""):
        """Creats folders and subfolders related to the F2 process in the
        folder _path_.

        Arguments
        ----
            model
                model of the network

            neuronal_network_path_extension
                When a string is specified, input, intermediateData and output
                folder will be stored in a subfolder with that name. If not,
                these folders will be stored directly in the
                path_neuronal_network_data folder.
        """

        self.model = model

        self.path_data = "data"
        self.path_neuronal_network_data = "/neuronal_network"

        if neuronal_network_path_extension != "":
            if neuronal_network_path_extension[-1] == '/':
                neuronal_network_path_extension = neuronal_network_path_extension[:-1]
            if neuronal_network_path_extension[0] != '/':
                neuronal_network_path_extension = "/" + neuronal_network_path_extension

        print(
            "path_neuronal_network_data: {}".format(
                self.path_data +
                self.path_neuronal_network_data))

        self.path_input = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/input"
        self.path_input_model = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/input/model"
        self.path_input_training_data = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/input/training_data"
        self.path_input_training_data_ground_truth = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/input/training_data/ground_truth"
        self.path_input_validation_data = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/input/validation_data"
        self.path_input_validation_data_ground_truth = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/input/validation_data/ground_truth"
        self.path_input_test_data = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/input/test_data"
        self.path_input_test_data_ground_truth = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/input/test_data/ground_truth"
        self.path_input_pretrained_weights = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/input/pretrained_weights"

        self.path_intermediate_data_trained_weights = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/intermediate_data/trained_weights"
        self.path_intermediate_data_history = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/intermediate_data/history"

        self.path_output_validation_data_prediction = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/output/validation_data_predictions"
        self.path_output_test_data_prediction = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/output/test_data_predictions"
        self.path_output_evaluation = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/output/evaluation"
        self.path_output_documentation = self.path_neuronal_network_data + \
            neuronal_network_path_extension + "/output/documentation"

        self.file_name_trained_weights = "weights.hdf5"
        self.file_name_predictions = "prediction.npy"
        self.file_name_history = "history.pkl"

        # create input folders
        os.makedirs(self.path_data + self.path_input, 0o777, True)
        os.makedirs(self.path_data + self.path_input_model, 0o777, True)
        os.makedirs(
            self.path_data +
            self.path_input_training_data,
            0o777,
            True)
        os.makedirs(
            self.path_data +
            self.path_input_training_data_ground_truth,
            0o777,
            True)
        os.makedirs(self.path_data + self.path_input_test_data, 0o777, True)
        os.makedirs(
            self.path_data +
            self.path_input_pretrained_weights,
            0o777,
            True)

        # create intermediate data folders
        os.makedirs(
            self.path_data +
            self.path_intermediate_data_trained_weights,
            0o777,
            True)
        os.makedirs(
            self.path_data +
            self.path_intermediate_data_history,
            0o777,
            True)

        # create output folders
        os.makedirs(
            self.path_data +
            self.path_output_test_data_prediction,
            0o777,
            True)
        os.makedirs(
            self.path_data +
            self.path_output_validation_data_prediction,
            0o777,
            True)
        os.makedirs(
            self.path_data +
            self.path_output_documentation,
            0o777,
            True)
        os.makedirs(self.path_data + self.path_output_evaluation, 0o777, True)

    def load_model(self, modelFilePath="",
                   neuronal_network_path_extensionPretrainedWeights=""):
        """
        Loads the model and copies the _modelFilePath_ into the input folder.

        # Argumets
            modelFilePath
                relative path to the model in parent _path_data_ folder

            neuronal_network_path_extensionPretrainedWeights
                name of the subfolder at _path_neuronal_network_data_ path,
                which will be used instead of _path_neuronal_network_data_

        Returns
    ----
            the model object
        """

        if modelFilePath != "":
            copyfile(
                modelFilePath,
                self.path_data +
                self.path_input_model +
                "/model.py")

        # model is defined in model.py
        self.model = get_model_deep_speckle()

        # load weights of the previous fog layer
        if (neuronal_network_path_extensionPretrainedWeights != ""):
            pos = self.path_intermediate_data_trained_weights.find(
                "/intermediateData")
            path = self.path_data \
                + self.path_neuronal_network_data \
                + "/" + neuronal_network_path_extensionPretrainedWeights \
                + self.path_intermediate_data_trained_weights[pos:] \
                + "//" + self.file_name_trained_weights

            self.load_weights(path)
#            copyfile(path, self.path_data + self.path_input_pretrained_weights + "/" + neuronal_network_path_extensionPretrainedWeights + "_" + self.file_name_trained_weights)
#            self.model.load_weights(path)

        return self.model

    def load_weights(self, path=""):
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
            path = self.path_data \
                    + self.path_intermediate_data_trained_weights \
                    + "/" + self.file_name_trained_weights
        else:
            copyfile(
                path,
                self.path_data
                + self.path_input_pretrained_weights
                + "/" + self.file_name_trained_weights)

        self.model.load_weights(path)

        return self.model

    def load_training_data(self, image_path, prefix=""):
        """
        Loads the training data and copies the listed files under
        _image_path_ into the input folder.

        Arguments
        ----
            image_path
                List of relative paths to the training data images in the
                parent folder _path_data_.

            prefix
                A prefix can be added to the image file during the copy
                operation.

        Returns
        ----
            a list of the copied images
        """
        path = toolbox.load_image(
            image_path,
            self.path_data +
            self.path_input_training_data,
            prefix=prefix)
        return path

    def load_ground_truth_data(self, image_path,
                               load_multiple_times_with_prefix=[]):
        """Loads the ground truth training data and copies the listed files
        under _image_path_ into the input folder.

        Arguments
        ----
            image_path
                List of relative paths to the ground truth training data images
                in the parent folder _path_data_.

            load_multiple_times_with_prefix
                If this is set, than all images listed under _image_path_ are
                saved multiple times with the listed prefix.

        Returns
        ----
            a list of the copied images
        """
        path = []
        if load_multiple_times_with_prefix == []:
            path = toolbox.load_image(
                image_path,
                self.path_data +
                self.path_input_training_data_ground_truth)
        else:
            for prefix in load_multiple_times_with_prefix:
                path += toolbox.load_image(
                    image_path,
                    self.path_data +
                    self.path_input_training_data_ground_truth,
                    prefix="{}_".format(prefix))
        return path

    def load_test_data(self, image_path, prefix=""):
        """Loads the test data and copies the listed files under _image_path_
        into the input folder.

        Arguments
        ----
            image_path
                List of relative paths to the test data images in the parent
                folder _path_data_.

            prefix
                A prefix can be added to the image file during the copy
                operation.

        Returns
        ----
            a list of the copied images
        """
        path = toolbox.load_image(
            image_path,
            self.path_data +
            self.path_input_test_data,
            prefix=prefix)
        return path

    def load_test_ground_truth_data(
            self, image_path, load_multiple_with_prefix=[]):
        """Loads the ground truth test data and copies the listed files under
        _image_path_ into the input folder.

        Arguments
        ----
            image_path
                List of relative paths to the ground truth test data images in
                the parent folder _path_data_.

            load_multiple_times_with_prefix
                If this is set, than all images listed under _image_path_ are
                saved multiple times with the listed prefix.

        Returns
        ----
            a list of the copied images
        """
        path = []
        if load_multiple_with_prefix == []:
            path = toolbox.load_image(
                image_path,
                self.path_data +
                self.path_input_test_data_ground_truth)
        else:
            for prefix in load_multiple_with_prefix:
                destination = self.path_data \
                                + self.path_input_test_data_ground_truth
                path += toolbox.load_image(
                        image_path,
                        destination,
                        prefix="{}_".format(prefix))
        return path

    def load_validation_data(self, image_path, prefix=""):
        """Loads the validation data and copies the listed files under
        _image_path_ into the input folder.

        Arguments
        ----
            image_path
                List of relative paths to the validataion data images in the
                parent folder _path_data_.

            prefix
                A prefix can be added to the image file during the copy
                operation.

        Returns
        ----
            a list of the copied images
        """
        path = toolbox.load_image(
            image_path,
            self.path_data +
            self.path_input_validation_data,
            prefix=prefix)
        return path

    def load_validation_ground_truth_data(
            self, image_path, load_multiple_with_prefix=[]):
        """Loads the ground truth test data and copies the listed files under
        _image_path_ into the input folder.

        Arguments
        ----
            image_path
                List of relative paths to the ground truth test data images in
                the parent folder _path_data_.

            load_multiple_times_with_prefix
                If this is set, than all images listed under _image_path_ are
                saved multiple times with the listed prefix.

        Returns
        ----
            a list of the copied images
        """
        path = []
        if load_multiple_with_prefix == []:
            path = toolbox.load_image(
                image_path,
                self.path_data +
                self.path_input_validation_data_ground_truth)
        else:
            for prefix in load_multiple_with_prefix:
                destination = self.path_data \
                                + self.path_input_validation_data_ground_truth
                path += toolbox.load_image(
                        image_path,
                        destination,
                        prefix="{}_".format(prefix))
        return path

    def get_image_as_npy(self, image_path):
        """ Opens the images listed under _image_path_ and returns them as a
        list of numpy arrays.

        Arguments
        ----
            image_path
                List of relative paths to the ground truth test data images in
                the parent folder _path_data_.
        
        Returns
        ----
            
        """
        rv = []
        for ip in image_path:
            file_extension = os.path.splitext(ip)[-1]

    #        base = os.path.basename(ip)
    #        name = os.path.splitext(base)
            data = []

            if file_extension == ".bmp":
                img = Image.open(ip)  # .convert('LA')
                img.load()
                data = np.asarray(img, dtype="int32") / 255
            elif (file_extension == ".npy") | (file_extension == ".bin"):
                data = np.load(ip)
                data = Image.fromarray(np.uint8(data * 255))  # .convert('LA')
                data = np.asarray(data, dtype="int32") / 255
            else:
                continue

            rv = rv + [data]

        rv = convert_image_list_to_4D_np_array(rv)

        return rv

    def save_4D_npy_as_bmp(self, npy_array, filename,
                           bmp_folder_path, invert_color=False):
        """Saves the 4d numpy array as bmp files.

        Dimensions:
            - 1d: image number
            - 2d: x dimension of the image
            - 3d: y dimension of the image
            - 4d: channel

        Arguments
        ----
            npy_array
                numpy array

            filename
                list of bmp filenames

            bmp_folder_path
                Destination where the bmp files will be stored.

            invert_color
                If it's True, than the colors of the images will be inverted.

        Returns
        ----
            the bmp file paths.
        """
        path = []
        for i in range(len(npy_array)):
            image = toolbox.convert_3d_npy_to_image(npy_array[i], invert_color)

            buffer = bmp_folder_path + "/{}.bmp".format(filename[i])
            image.save(buffer)

            path += [buffer]

        return path

    def train_network(self, training_data_path,
                      ground_truth_path, fit_epochs, fit_batch_size):
        """ Runs the routine to train the network.

        Arguments
        ----
            training_data_path
                list of paths to images

            ground_truth_path
                list of paths to images

            fit_epochs
                number of epochs

            fit_batch_size
                batch size
        """
        training_data_path.sort()
        ground_truth_path.sort()

        training_data = self.get_image_as_npy(training_data_path)
        ground_truth = self.get_image_as_npy(ground_truth_path)

        # Compile model
        adam = keras.optimizers.Adam(
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.001 /
            fit_epochs,
            amsgrad=False)
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=adam)

        # Fit the model
        history = self.model.fit(training_data,
                                 ground_truth,
                                 epochs=fit_epochs,
                                 batch_size=fit_batch_size,
                                 verbose=2)
# todo:
#        history = model.fit_generator(generate_arrays(training_data_path, ground_truth_path, fit_batch_size),
#                      steps_per_epoch=fit_batch_size, epochs=fit_epochs)

        # saving the history
        with open(self.path_data + self.path_intermediate_data_history + "/"
                  + self.file_name_history, "wb") as f:
            pickle.dump(history.history, f)

        self.model.save_weights(
                self.path_data
                + self.path_intermediate_data_trained_weights
                + "/" + self.file_name_trained_weights)

    def validate_network(self, validation_data_path, trained_weights_path=""):
        """Calculates predictions based on the validation dataset.

        Arguments
        ----
            validation_data_path
                list of paths to images

            trained_weights_path
                If a path is specified, the model uses these weights instead
                of the weights from the intermediateData folder.

        Returns
        ----
            the path of the prediction.
        """
        path = self.predict(
            validation_data_path,
            self.path_data +
            self.path_output_validation_data_prediction,
            trained_weights_path)

        return path

    def test_network(self, test_data_path, trained_weights_path=""):
        """Calculates predictions based on the test dataset.

        Arguments
        ----
            test_data_path
                list of paths to images

            trained_weights_path
                If a path is specified, the model uses these weights instead
                of the weights from the intermediateData folder.

        Returns
        ----
            the path of the prediction.
        """
        path = self.predict(
            test_data_path,
            self.path_data +
            self.path_output_test_data_prediction,
            trained_weights_path)

        return path

    def predict(self, data_path, prediction_folder_path,
                trained_weights_path=""):
        """Calculates pridictions based on the given data_path.

        Arguments
        ----
            data_path
                path of the input data

            prediction_folder_path
                folder where the prediction shoud be stored

            trained_weights_path
                If a path is specified, the model uses these weights instead
                of the weights from the intermediateData folder.

        Returns
        ----
            the path of the prediction.
        """
        data_path.sort()
        test_data = self.get_image_as_npy(data_path)

        fileName = []
        for ip in data_path:
            base = os.path.basename(ip)
            fileName += [os.path.splitext(base)[0]]

        if trained_weights_path == "":
            self.model.save_weights(
                self.path_data +
                self.path_intermediate_data_trained_weights +
                "/" +
                self.file_name_trained_weights)

        pred = self.model.predict(test_data)

        path = self.save_4D_npy_as_bmp(
            pred, fileName, prediction_folder_path, invert_color=True)

        return path


    def jaccard_index(self, path_ground_truth, path_prediction, parameter = []):
        """Calculates the Jaccard index by comparing the images in both paths
        and stores them in a csv file. Optionally these values can be saved as
        an image (export_images).
        
        Arguments
        ----
            path_ground_truth
                list of paths for the ground truth dataset
            
            path_prediction
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

        pred = self.get_image_as_npy(path_prediction)
        ground_truth = self.get_image_as_npy(path_ground_truth)

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
        path_ji= self.path_output_evaluation + "/jaccard_index"
        path_ji_images = self.path_output_evaluation + "/jaccard_index/images"
        os.makedirs(self.path_data + path_ji, 0o777, True)
        
        image_file_name = []
        for i in range(len(path_prediction)):
            image_file_name += [toolbox.get_file_name(path_prediction[i])]
        image_file_name = np.stack(image_file_name)
        
        file = self.path_data + path_ji + "/" + file_name
        with open(file, 'w') as f:
            f.write("object name;jaccard index\n")
            f.write("mean;{}\n".format(meanNP))
            for i in range(len(meanNP_per_image)):
                f.write("{};{}\n".format(image_file_name[i], meanNP_per_image[i]))

        # ji as images
        if export_images is True:
            os.makedirs(self.path_data + path_ji_images, 0o777, True)
            for i in range(len(scoreNP[:,0,0])):
                image = toolbox.convert_3d_npy_to_image(scoreNP[i,:,:], invert_color=False)
    
                buffer = self.path_data + path_ji_images + "/{}.bmp".format(image_file_name[i])
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
            path_ground_truth
                list of paths for the ground truth dataset
            
            path_prediction
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

        pred = self.get_image_as_npy(path_prediction)
        ground_truth = self.get_image_as_npy(path_ground_truth)
        
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
        path_pcc= self.path_output_evaluation + "/pearson_correlation_coefficient"
        os.makedirs(self.path_data + path_pcc, 0o777, True)

        image_file_name = []
        for i in range(len(path_prediction)):
            image_file_name += [toolbox.get_file_name(path_prediction[i])]
        image_file_name = np.stack(image_file_name)
        
        file = self.path_data + path_pcc + "/" + file_name
        with open(file, 'w') as f:
            f.write("object name;pearson correlation coefficient\n")
            f.write("mean;{}\n".format(mean_pcc))
            for i in range(len(value_pcc)):
                f.write("{};{}\n".format(image_file_name[i], value_pcc[i]))

            
    
    def evaluate_network(self, method, path_ground_truth, path_prediction,
                         additional_method_parameter = []):
        """Wrapper function to call multiple evaluation methods
        
        Arguments
        ----
            method
                list of functions, but at least it can be a singel function

            path_ground_truth
                list of paths for the ground truth dataset
            
            path_prediction
                list of paths for the prediction dataset
        
        Returns 
        ----
            list of method returns, if any.
        """
        path_ground_truth.sort()
        path_prediction.sort()

        rv = []

        if type(method) is list:
            for i in range(len(method)):
                if additional_method_parameter != []:
                    rv += [method[i](path_ground_truth, path_prediction, additional_method_parameter[i])]
                else:
                    rv += [method[i](path_ground_truth, path_prediction)]
        else:
            if additional_method_parameter != []:
                rv = [method(path_ground_truth, path_prediction, additional_method_parameter)]
            else:
                rv = [method(path_ground_truth, path_prediction)]


#        for i in range(len(scoreNP[:,0,0])):

#        pred_tf = tf.convert_to_tensor(pred)
#        ground_truth_tf = tf.convert_to_tensor(ground_truth)

#        rv = []
#
#        rv += [pcc(ground_truth_tf, pred_tf)]

#        for m in method:
#            sess = tf.InteractiveSession()
#
#            score = m(ground_truth, pred)
#            mean = K.mean(score)
#            rv += [mean.eval()]

#        return rv

    def loadHistory():
        with open('history.pkl', 'rb') as handle:
            hist = pickle.load(handle)
