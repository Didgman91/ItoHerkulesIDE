#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:26:02 2018

@author: daiberma
"""

import os

import numpy as np

from PIL import Image
from PIL import ImageOps

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
    if (isinstance(images, list)) & (len(images) > 1):
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
                buffer += [np.asarray(b, dtype="int32")]

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
    image = Image.fromarray(np.uint8(npy*255)).convert('RGB')
    if invert_color is True:
        image = ImageOps.invert(image)

    return image