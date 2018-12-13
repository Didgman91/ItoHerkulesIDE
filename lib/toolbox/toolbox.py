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


def send_message(message):
    subprocess.Popen(['notify-send', message])
    return


def get_file_path_with_extension(path_folder, extension):
    """Scans a folder for all files with a specific extension.

    Arguments
    ----
        path_folder
            search folder
        extension
            accepts a list of file extensions

    Returns
    ----
        list with all files in the _path_folder_ with a specific _extension_
    """
    if path_folder[len(path_folder)-1] != "/":
        path_folder = path_folder + "/"

    file_name = os.listdir(path_folder)

    filePath = []

    for name in file_name:
        file_extension = os.path.splitext(name)[1]
        for ex in extension:
            if ex[0] != ".":
                ex = "." + ex
            if (file_extension == ex):
                filePath = filePath + [path_folder + name]

    filePath.sort()

    return filePath


def get_file_path_with_extension_include_subfolders(path_folder, extension):
    """returns a list with all files in the _path_folder_ with a specific
    _extension_

    Arguments
    ----
        path_folder
            search folder
        extension
            accepts a list of file axtensions
    """

    if path_folder[len(path_folder)-1] != "/":
        path_folder = path_folder + "/"

    filePath = []

    for root, dirs, files in os.walk(path_folder):
        print(root)
        print("dirs: {}".format((len(dirs))))
        print("files: {}\n".format(len(files)))

        for name in files:
            file_extension = os.path.splitext(name)[1]
            for ex in extension:
                if ex[0] != ".":
                    ex = "." + ex
                if (file_extension == ex):
                    if root[len(root)-1] != "/":
                        root += "/"
                    filePath = filePath + [root + name]

    filePath.sort()

    return filePath


def get_file_name(path):
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
    file_name = os.path.splitext(base)[0]

    return file_name


def load_image(source_path, destination_path, invert_color=False,
               resize=False, x_pixel=0, y_pixel=0, prefix=""):
    """Loads images and saves them in the path _destination_path_.

    Arguments
    ----
        source_path
            list of images paths.
        destination_path
            folder in which  the image is to be saved
        invert_colors
            inverts the color of the image

        resize
            if this is true, the image will be resized with _x_pixel_
            and _y_pixel_ parameters.
        x_pixel
            number of pixels in x direction, if the image is to be resized
        y_pixel
            number of pixels in y direction, if the image is to be resized
        prefix
            adds an optional prefix to the file_name

    Retruns
    ----
        list of image paths in the _destination_path_
    """
    os.makedirs(destination_path, 0o777, True)

    rv = []
    for ip in source_path:
        file_extension = os.path.splitext(ip)[-1]

        im = Image.open(ip)

        im = im.convert("RGB")  # 24 bit: required by F2

        if resize is True:
            im = im.resize((x_pixel, y_pixel))

        if invert_color is True:
            im = ImageOps.invert(im)

        base = os.path.basename(ip)
        name = os.path.splitext(base)

        if destination_path[-1] != "/":
            destination_path += "/"

        if prefix == "":
            path = destination_path + name[0] + file_extension
        else:
            path = destination_path + prefix + name[0] + file_extension

        im.save(path)
        rv = rv + [path]

    return rv


def load_np_images(path_image, extension=["npy", "bin"]):
    """Loads a list of numpy images and returns it.

    Arguments
    ----
        path_image
            path list of numpy images
        exteinsion
            specifies the file extension

    Returns
    ----
        Lsit of numpy arrays that contain the images.
    """
    image = []

    for i in path_image:
        file_extension = os.path.splitext(i)[-1]
        for ex in extension:
            if ex[0] != ".":
                ex = "." + ex
            if (file_extension == ex):
                image = image + [np.load(i)]

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


def run_process(process, arg=[""]):
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
        exit_code
            exit_code of the process
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
    exit_code = p.returncode

    exit_code = p.wait()
    print("exit_code: {}".format(exit_code))
    sys.stdout.flush()
    t = time.time() - start

    output_decoded = []
    for i in range(len(output)-1):
        output_decoded = output_decoded + [output[i].decode("utf-8")]

    return output_decoded, exit_code, t
