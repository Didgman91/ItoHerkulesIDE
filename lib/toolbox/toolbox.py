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

import errno

import re

import fileinput

import numpy as np

from PIL import Image
from PIL import ImageOps

import shutil

from lib.toolbox.zipData import zip_data

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


def get_file_name(path, with_extension = False):
    """ Returns the file name out of the path.

    Arguments
    ----
        path
            path to file

        with_extension
            if it is set to True, the return value also contains the extension.

    Returns
    ----
        the file name
    """
    base = os.path.basename(path)
    file_name = os.path.splitext(base)[0]
    
    if with_extension is True:
        file_name += os.path.splitext(base)[1]

    return file_name

def get_file_extension(path):
    """ Returns the file extension out of the path.

    Arguments
    ----
        path
            path to file

    Returns
    ----
        the file extension
    """
    base = os.path.basename(path)
    file_extension = os.path.splitext(base)[1]

    return file_extension

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

def make_distinct_list(seq):
    """Makes a list of distinct elements.
    Arguments
    ----
        seg: list
            input list
    
    Returns
    ----
        a list of distinct elements. The order is not preserved.
    """
    return list(set(seq))

def copy(source, destination, *ignore_patterns):
    """ Copies entire folders.
    
    Arguments
    ----
        source
            Folder to copy.
        destination
            Folder in which the copy is stored.
    Reference
    ----
        https://www.pythoncentral.io/how-to-recursively-copy-a-directory-folder-in-python/
    """
    try:
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(ignore_patterns))
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(source, destination)
        else:
            print('Directory not copied. Error: %s' % e)

def copy_folder(source, destination):
    """ Copies entire folders.
    
    Arguments
    ----
        source
            Folder to copy.
        destination
            Folder in which the copy is stored.
    Reference
    ----
        https://www.pythoncentral.io/how-to-recursively-copy-a-directory-folder-in-python/
    """
    try:
        shutil.copytree(source, destination)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

def create_folder(path):
    """makes a directory
    
    Argument
    ----
        path
            path of the directory
    """
    os.makedirs(path, 0o777, True)

def replace(text, dictionary):
    """Replaces placeholders in a text with a dictionary.
    
    Arguments
    ----
        text
            string with placeholders
        dictionary
            Dictionary used to replace placeholders.

    Returns
    ----
        a text without placeholders, if the dictonary contains all
        placeholders.
    """
    pattern = re.compile(r'\b(' + '|'.join(dictionary.keys()) + r')\b')
    result = pattern.sub(lambda x: dictionary[x.group()], text)
    
    return result

def replace_in_file(path, dictionary):
    """Replaces placeholders in a text file with a dictionary.

    Arguments
    ----
        path
            text file path
        dictionary
            Dictionary used to replace placeholders.
    """
    with fileinput.FileInput(path, inplace=True) as file:
        for line in file:
            buffer = replace(line, dictionary)
            print(buffer, end='')

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


def run_process(process, arg=[""], working_dir = "", path_stdout_file = ""):
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
    t = time.strftime("%y%m%d_%H%M")
#    arg += ["| tee " + path_stdout_file + "{}_stdout.txt".format(t)]
    p = subprocess.Popen([process] + arg,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         cwd=os.getcwd() + working_dir)

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

def backup(executed_modules, path_backup="backup"):
    create_folder(path_backup)
    
    modules = ""
    folders = []
    for m in executed_modules:
        modules += "_" + m
        folders += ["data/" + m]
    
    folders = make_distinct_list(folders)
    
    t = time.strftime("%y%m%d_%H%M")
    zip_settings = {'zip_file_name': "{}/{}{}.zip".format(path_backup, t, modules),
                    'zip_include_folder_list': ["config", "lib"] + folders,
                    'zip_include_file_list': ["main.py"],
                    'skipped_folders': [".git", "__pycache__"]}
    
    flag_copy = True
    
    def copy_files(data, destination):
        for f in data:
            copy(f, "{}/{}{}".format(path_backup, t, modules))
    
    if flag_copy is True:
        create_folder("{}/{}{}".format(path_backup, t, modules))
        copy_files(zip_settings["zip_include_folder_list"] + zip_settings["zip_include_file_list"])
    else:
        zip_data(zip_settings)
        
def save_as_csv(array, path, header):
    """
    Arguments
    ----
        array: numpy array
            1- or 2-dimensional array
        path: string
            path where the csv file is stored
        header: list<string>
            list of the headers for each colmn of the csv file
    """
    array_shape = np.shape(array)
    
    export = []
    if len(array_shape) == 1:
        for ii in range(len(array)):
            export += [np.array([ii+1, array[ii]])]
    elif len(array_shape) == 2:
        export = array
    elif len(array_shape) > 2:
        return 
    
    delemiter = ","
    header_intern = ""
    for h in header:
        header_intern += h + delemiter
    
    header_intern = header_intern[:-len(delemiter)]
    
    np.savetxt(path,
               export,
               delimiter = delemiter,
               header=header_intern,
               comments='')