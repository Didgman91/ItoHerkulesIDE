#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:54:05 2018

@author: maxdh
"""

import os
import sys
import subprocess
import time

import numpy as np

from PIL import Image, ImageOps


def send_Message(message):
    subprocess.Popen(['notify-send', message])
    return


def get_file_path_with_extension(path_Folder, extension):
    """Scans a folder for all files with a specific extension.

    Arguments
    ----
        path_Folder
            search folder
        extension
            accepts a list of file extensions

    Returns
    ----
        list with all files in the _path_Folder_ with a specific _extension_
    """
    if path_Folder[len(path_Folder)-1] != "/":
        path_Folder = path_Folder + "/"

    file_Name = os.listdir(path_Folder)

    filePath = []

    for name in file_Name:
        file_Extension = os.path.splitext(name)[1]
        for ex in extension:
            if ex[0] != ".":
                ex = "." + ex
            if (file_Extension == ex):
                filePath = filePath + [path_Folder + name]

    filePath.sort()

    return filePath


def get_file_path_with_extension_include_subfolders(path_Folder, extension):
    """returns a list with all files in the _path_Folder_ with a specific
    _extension_

    Arguments
    ----
        path_Folder
            search folder
        extension
            accepts a list of file axtensions
    """

    if path_Folder[len(path_Folder)-1] != "/":
        path_Folder = path_Folder + "/"

    filePath = []

    for root, dirs, files in os.walk(path_Folder):
        print(root)
        print("dirs: {}".format((len(dirs))))
        print("files: {}\n".format(len(files)))

        for name in files:
            file_Extension = os.path.splitext(name)[1]
            for ex in extension:
                if ex[0] != ".":
                    ex = "." + ex
                if (file_Extension == ex):
                    if root[len(root)-1] != "/":
                        rootBuffer = root + "/"
                    filePath = filePath + [rootBuffer + name]

    filePath.sort()

    return filePath


def get_File_Name(path):
    """ Returns the file name out of the path.

    Arguments
    ----
        path
            path to file

    Returns
    ----
        the file name
    """
    base = os.path.basename(path)
    file_Name = os.path.splitext(base)[0]

    return file_Name


def load_Image(source_Path, destination_Path, invert_Color=False,
               resize=False, x_Pixel=0, y_Pixel=0, prefix=""):
    """Loads images and saves them in the path _destination_Path_.

    Arguments
    ----
        source_Path
            list of images paths.
        destination_Path
            folder in which  the image is to be saved
        invert_Colors
            inverts the color of the image

        resize
            if this is true, the image will be resized with _x_Pixel_
            and _y_Pixel_ parameters.
        x_Pixel
            number of pixels in x direction, if the image is to be resized
        y_Pixel
            number of pixels in y direction, if the image is to be resized
        prefix
            adds an optional prefix to the file_Name

    Retruns
    ----
        list of image paths in the _destination_Path_
    """
    os.makedirs(destination_Path, 0o777, True)

    rv = []
    for ip in source_Path:
        file_Extension = os.path.splitext(ip)[-1]

        im = Image.open(ip)

        im = im.convert("RGB")  # 24 bit: required by F2

        if resize is True:
            im = im.resize((x_Pixel, y_Pixel))

        if invert_Color is True:
            im = ImageOps.invert(im)

        base = os.path.basename(ip)
        name = os.path.splitext(base)

        if destination_Path[-1] != "/":
            destination_Path += "/"

        if prefix == "":
            path = destination_Path + name[0] + file_Extension
        else:
            path = destination_Path + prefix + name[0] + file_Extension

        im.save(path)
        rv = rv + [path]

    return rv


def load_np_images(path_Image, extension=["npy", "bin"]):
    """Loads a list of numpy images and returns it.

    Arguments
    ----
        path_Image
            path list of numpy images
        exteinsion
            specifies the file extension

    Returns
    ----
        Lsit of numpy arrays that contain the images.
    """
    image = []

    for i in path_Image:
        file_Extension = os.path.splitext(i)[-1]
        for ex in extension:
            if ex[0] != ".":
                ex = "." + ex
            if (file_Extension == ex):
                image = image + [np.load(i)]

    return image


def convert_3d_Npy_To_Image(npy, invert_Color=False):
    """Converts a numpy array to an image.

    Arguments
    ----
        npy
            3 dimensional numyp array

        invert_Color
            If this is true, the colors in the image will be inverted.

    Returns
    ----
        the image
    """
    image = Image.fromarray(np.uint8(npy*255)).convert('RGB')
    if invert_Color is True:
        image = ImageOps.invert(image)

    return image


def print_program_section_name(name):
    """Formats and prints the _name_ on stdout.

    Arguments
    ----
        name
            string to be printed
    """

    line = ""
    length = 45
    if length < len(name) + 5:
        length = len(name) + 5

    for i in range(length):
        line = line + "-"

    print("# {}".format(line))
    print("# {}".format(name))
    print("# {}".format(line))


def run_Process(process, arg=[""]):
    """stars a process with arguments

    Arguments
    ----
        process
            string contains the path or directly die programm name
            which to be excecuted
        arg
            list of arguments

    Returns
    ----
        output
            list of strings of the stream stdout
        exit_Code
            exit_Code of the process
        t
            wall time of the process [s]
    """

    start = time.time()

    print("----- Start Subprocess ----")
    print("Process: {}".format(process))
    print("Argument: {}".format(arg))
    sys.stdout.flush()
    p = subprocess.Popen([process] + arg,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         cwd=os.getcwd())

    output = p.communicate()
    exit_Code = p.returncode

    exit_Code = p.wait()
    print("exit_Code: {}".format(exit_Code))
    sys.stdout.flush()
    t = time.time() - start

    output_Decoded = []
    for i in range(len(output)-1):
        output_Decoded = output_Decoded + [output[i].decode("utf-8")]

    return output_Decoded, exit_Code, t
