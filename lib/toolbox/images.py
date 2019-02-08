#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:26:02 2018

@author: daiberma
"""

import os

import numpy as np
from scipy import signal

from PIL import Image
from PIL import ImageOps

import imageio

def get_image_as_npy(image_path):
    """ Opens the images listed under _image_path_ and returns them as a
    list of numpy arrays.

    Arguments
    ----
        image_path
            List of relative paths to the ground truth test data images in
            the parent folder _path_data_.

    Returns
    ----
        the image as a numpy array.
    """
    rv = []

    if isinstance(image_path, list) is False:
        image_path = [image_path]

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
    if (isinstance(images, list)) & (len(images) > 0):
        # stack a list to a numpy array
        images = np.stack((images))
    elif isinstance(images, np.ndarray):
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
                buffer += [np.asarray(b)/255]

            images = np.stack((buffer))
    if images.max() > 1:
        images = images / 255

    # reshape the numpy array (imageNumber, xPixels, yPixels, 1)
    if len(images.shape) == 3:
        images = images.reshape(
            (images.shape[0], images.shape[1], images.shape[2], 1))

    return images

def save_4D_npy_as_bmp(npy_array, filename,
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
        image = convert_3d_npy_to_image(npy_array[i], invert_color)

        buffer = bmp_folder_path + "/{}.bmp".format(filename[i])
        image.save(buffer)

        path += [buffer]

    return path


def convert_3d_npy_to_image(npy, invert_color=False):
    """Converts a numpy array to an image.

    Arguments
    ----
        npy
            3 dimensional numyp array

        invert_color
            If this is true, the colors in the image will be inverted.

    Returns
    ----
        the image
    """
    image = []
    if np.shape(npy)[-1] == 1:
        image = Image.fromarray(np.uint8(npy*255)[:,:,0], 'L').convert('RGB')
    else:
        image = Image.fromarray(np.uint8(npy*255)).convert('RGB')

    if invert_color is True:
        image = ImageOps.invert(image)

    return image

def get_convolution_2d(image_1, image_2):
    """Calculates the convolution of two images using signal.fftconvolve().
    
    Arguments
    ----
        image_1
            first four-dimensional input array
        image_2
            second four-dimensional input array

    Returns
    ----
        an array with the convolution of image_1 with image_2

    Dimensions
    -----
        first dimension
            image number
        second dimension
            first dimension of the image
        third dimension
            second dimension of the image
        fourth dimension
            channel of the image

        The convolution is calculated for each dataset and channel.
    """
    buffer = []
    shape = np.shape(image_1)
    dimension_image_number = 0
    dimension_iamge_channel = 3
    
    for i in range(shape[dimension_image_number]):
        buffer_channel = []
        for ii in range(shape[dimension_iamge_channel]):
            buffer_channel += [signal.fftconvolve(image_1[i,:,:,ii],
                                                  image_2[i,:,:,ii])]
        buffer_channel = np.stack(buffer_channel, axis=2)
        buffer += [buffer_channel]
    
    buffer = np.stack(buffer)
    
    return buffer

def get_cross_correlation_2d(image_1, image_2):
    """Calculates the cross-correlation of two images using
    signal.fftconvolve().
    
    Arguments
    ----
        image_1
            first four-dimensional input array
        image_2
            second four-dimensional input array

    Returns
    ----
        an array with the cross-correlation of image_1 with image_2

    Dimensions
    -----
        first dimension
            image number
        second dimension
            first dimension of the image
        third dimension
            second dimension of the image
        fourth dimension
            channel of the image

        The cross-correlation is calculated for each dataset and channel.
    """
    buffer = []
    shape = np.shape(image_1)
    dimension_image_number = 0
    dimension_image_channel = 3
    
    if len(shape) == 2:
        buffer = signal.fftconvolve(image_1,image_2[::-1,::-1])
    elif len(shape) == 4:
        for i in range(shape[dimension_image_number]):
            buffer_channel = []
            for ii in range(shape[dimension_image_channel]):
                buffer_channel += [signal.fftconvolve(image_1[i,:,:,ii],
                                                      image_2[i,:,:,ii][::-1,::-1])]
            buffer_channel = np.stack(buffer_channel, axis=2)
            buffer += [buffer_channel]
        
        buffer = np.stack(buffer)
    
    return buffer

def get_autocorrelation_2d(image_1):
    """Calculates the autocorrelation of two images using
    signal.fftconvolve().
    
    Arguments
    ----
        image_1
            first four-dimensional input array
        image_2
            second four-dimensional input array

    Returns
    ----
        an array with the autocorrelation of image_1 with image_2

    Dimensions
    -----
        first dimension
            image number
        second dimension
            first dimension of the image
        third dimension
            second dimension of the image
        fourth dimension
            channel of the image

        The autocorrelation is calculated for each dataset and channel.
    """
    buffer = get_cross_correlation_2d(image_1, image_1)
    
    return buffer

def get_intersection(array_2d, intersection_point=[], axis = 0):
    """Returns the intersection of a 2d array. If no middle value is given, the
    
    Arguments
    ----
        array_2d: array
            two-dimensional array
        middle: list(integer)
            contains two values representing the intersection point
    
    Returns
    ----
        a vector e.g. array_2d[0,:]
    """
    if intersection_point == []:
        size = np.array(np.shape(array_2d))
        intersection_point = (size/2).astype(int)
    
    if axis == 0:
        return array_2d[:,intersection_point[1]]
    else:
        return array_2d[intersection_point[0],:]

def __get_max_position_2d(array_2d):
    """
    Argument
    ----
        array_2d: np.2darray
            two-dimensional array, but it can also be a one-dimensional array.
    Returns
    ----
        the indices of the maximum values of a two-dimensional array.
        If there are several items with the same maximum value, the middle
        position of these items is calculated.
        
        position_max: np.2darray
        
        std: stadnard deviation tupel [x, y]
    """
    
    buffer_max = np.where(array_2d == array_2d.max())
    
    buffer_max_x = [buffer_max[0].min(), buffer_max[0].max()]
    buffer_max_y = [buffer_max[1].min(), buffer_max[1].max()]
    
    
    posistion_max = np.array([(buffer_max_x[1] + buffer_max_x[0]) / 2,
                         (buffer_max_y[1] + buffer_max_y[0]) / 2])
    
    std_x = np.std(buffer_max[0])
    std_y = np.std(buffer_max[1])
    
    std = np.array([std_x, std_y])
    
    return posistion_max, std

def get_max_position(image, relative_position=[]):
    """ Returns the indices of the maximum value of an array.
    
    **!!! Note the changed dimension meaning of the return value !!!**
    
    Accepted dimensionality
    ----
        Up to four-dimensional datasets are accepted.
        
        - one dimension
            series of measuring points
        - two dimensions
            two-dimensional array (e.g. an image with just
            intensity values)
        - three dimensions
            an image with multiple channels
            ([x-pixles, y-pixels, color channels])
        - four dimensions
            a serie of images with multiple channels
            ([image number, x-pixles, y-pixels, color channels])

    Returns
    ----
        Depending on the _image_ dimension, it returns a scalar or a deep
        list.
        
        - one dimension
            The function returns a scalar.
        - two dimensions
            The function returns an array with the x- and y-position of the
            maximum in the image.
        - three dimensions
            The function returns a list with arrays. This contains the x- and
            y-position of the maximum. (e.g. [channel, x position, y position])
        - four dimensions
            The function returns a list in a list with array's. This contains
            the x- and y-position of the maximum.
            (e.g. [image number, channel, x position, y position])
    """
    def check_ndarray(data):
        if type(data) is np.ndarray:
            return True
        else:
            return False
    
    def calculate_relative_position(data, position):
        if len(np.shape(data)) == 1:
            if check_ndarray(data) is True:
                data -= position

        return data

    shape = np.array(np.shape(image))

    position_max = []
    std = []
    if len(shape) == 1:
        position_max, std = __get_max_position_2d(image)
        if relative_position != []:
                position_max = calculate_relative_position(position_max,
                                                     relative_position)

    elif len(shape) == 2:
        position_max, std = __get_max_position_2d(image)
        if relative_position != []:
                position_max = calculate_relative_position(position_max,
                                                     relative_position)

    elif len(shape) == 3:
        dimension_iamge_channel = 2
        for i in range(shape[dimension_iamge_channel]):
            buffer_pos, buffer_std = __get_max_position_2d(image[:,:,i])
            if relative_position != []:
                buffer = calculate_relative_position(buffer_pos,
                                                     relative_position)
            position_max += [buffer_pos]
            std += [buffer_std]
        np.stack(position_max)

    elif len(shape) == 4:
        dimension_image_number = 0
        dimension_iamge_channel = 3
        
        for i in range(shape[dimension_image_number]):
            buffer_position_max = []
            buffer_std = []
            for ii in range(shape[dimension_iamge_channel]):
#                buffer_position_max += [__get_max_position_2d(image[i,:,:,ii])]
                buffer, buffer_std_rv = __get_max_position_2d(image[i,:,:,ii])
                if relative_position != []:
                    buffer = calculate_relative_position(buffer,
                                                         relative_position)
                buffer_position_max += [buffer]
                buffer_std += [buffer_std_rv]
            buffer_position_max = np.stack(buffer_position_max)
            buffer_std = np.stack(buffer_std)
            position_max += [buffer_position_max]
            std += [buffer_std]
        position_max = np.stack(position_max)
        std = np.stack(std)

    return position_max, std

def get_histogram(image_path, bins=10):
    """ The histogram is computed over the flattened array.
    Arguments
    ----
        image_path: list<string>
            path of the image
        bins: int or sequence of scalars or str, optional
            further information: numpy.histogram
    
    Returns
    ----
        the histogram of the image per channel.
    """
    
    histogram = []
    for ip in image_path:
        image = imageio.imread(ip, 'bmp')
        
        image_shape = np.shape(image)
        
        hist = []
        if len(image_shape) == 3:
            for i in range(image_shape[2]):
                buffer, bin_edges = np.histogram(image, bins)
                hist += [buffer]
        else:
            hist, bin_edges = np.histogram(image, bins)
        
        histogram += [hist]
    
    return histogram