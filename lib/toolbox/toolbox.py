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
import random

import re

import fileinput

import numpy as np

import matplotlib.pyplot as plt

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
        path: [list<string>, string]
            path to file

        with_extension
            if it is set to True, the return value also contains the extension.

    Returns
    ----
        file_name: [list<string>, string]
            the file name
    """
    def __get_name(path):
        base = os.path.basename(path)
        file_name = os.path.splitext(base)[0]
        
        if with_extension is True:
            file_name += os.path.splitext(base)[1]
        
        return file_name

    file_name = []
    if type(path) is str:
        file_name = __get_name(path)
    elif type(path) is list:
        for p in path:
            file_name += [__get_name(p)]

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
        prefix: [string, list<string>]
            adds an optional prefix to the file_name

    Retruns
    ----
        list of image paths in the _destination_path_
    """
    os.makedirs(destination_path, 0o777, True)

    rv = []
    for i in range(len(source_path)):
        print("load image: {}".format(source_path[i]))
        
        file_extension = os.path.splitext(source_path[i])[-1]

        im = Image.open(source_path[i])

        im = im.convert("RGB")  # 24 bit: required by F2

        if resize is True:
            im = im.resize((x_pixel, y_pixel))

        if invert_color is True:
            im = ImageOps.invert(im)

        base = os.path.basename(source_path[i])
        name = os.path.splitext(base)

        if destination_path[-1] != "/":
            destination_path += "/"

        if type(prefix) is str:
            path = destination_path + prefix + name[0] + file_extension
        elif type(prefix) is list:
            if len(prefix) == len(source_path):
                path = destination_path + prefix[i] + name[0] + file_extension
        else:
            path = destination_path + name[0] + file_extension

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

def copy(source, destination, replace=False, ignore_patterns=[]):
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
        if replace is True:
            shutil.rmtree(destination, ignore_errors=True)
        
        if ignore_patterns == []:
            shutil.copytree(source, destination)
        else:
            ignore = shutil.ignore_patterns(ignore_patterns)
            shutil.copytree(source, destination, ignore=ignore)
        
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

def get_subfolders(path, show_hidden_folders = False):
    """
    Argument
    ----
        path: string
            path of a directory
        show_hidden_folders: bool
            if true, also the hidden subdirectories are returned..
    
    Returns
    ----
        a list<string> with the subdirectories.
    """
    content = os.listdir(path)
    output = [dI for dI in content if os.path.isdir(os.path.join(path,dI))]
    if show_hidden_folders is True:
        buffer = []
        for o in output:
            if o[0] != ".":
                buffer += [o]
        output = buffer
    return output

def read_file_lines(path):
    """
    Argument
    ----
        path: string
            path to a text file
    
    Returns
    ----
        the lines of a text file.
    """

    textFile = open(path, "r")
    lines = textFile.read().splitlines()
    
    return lines

def get_intersection(list_1, list_2, str_diff_exact=True):
    """
    Arguments
    ----
        list_1: list
            1-dimensional list of objects
        lsit_2: list
            1-dimensional list of objects
        str_diff_exact: boolean, optinal
            If *False* and the lists are lists of strings, then the string of a
            *list_2* element can be a substring of a *list_1* element. This is
            then also taken into account at the intersection of bouth lists.
    Returns
    ----
        a list with the intersection of *list_2* and *list_1*.
    """
    # check input
    if list_1 == [] or list_2 == [] or type(list_1) is not list or type(list_1) is not list:
        return
    
    def _intersection(first, second):
        second = set(second)
        return [item for item in first if item in second]
    
    # creat list with relative complement
    if str_diff_exact is False:
        if type(list_1[0]) is str and type(list_2[0]) is str:
            buffer = []
            for b in list_2:
                for a in list_1:
                    if b in a:
                        buffer += [a]
                    else:
                        continue
            return buffer
        else:
            _intersection(list_1, list_2)
            return
    else:
        return _intersection(list_1, list_2)

def get_relative_complement(list_1, list_2, str_diff_exact=True):
    """
    Arguments
    ----
        list_1: list
            1-dimensional list of objects
        lsit_2: list
            1-dimensional list of objects
        str_diff_exact: boolean, optinal
            If *True* and the lists are lists of strings, then the string of a
            *list_2* element can be a substring of a *list_1* element. This is then
            also removed from list *list_1*.
    Returns
    ----
        a list with the relative complement of *list_2* in *list_1*.
    """
    # check input
    if list_1 == [] or list_2 == [] or type(list_1) is not list or type(list_1) is not list:
        return
    
    def _diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]
    
    # creat list with relative complement
    if str_diff_exact is False:
        if type(list_1[0]) is str and type(list_2[0]) is str:
            for b in list_2:
                buffer = []
                for a in list_1:
                    if b in a:
                        continue
                    else:
                        buffer += [a]
                list_1 = buffer
            return list_1
        else:
            _diff(list_1, list_2)
            return
    else:
        return _diff(list_1, list_2)
    

def replace(text, dictionary):
    """Replaces placeholders in a text with a dictionary.
    
    Arguments
    ----
        text
            string with placeholders
        dictionary
            Dictionary used to replace placeholders.

    Dictionary
    -----
        Example Dictionary:
            >>> dictionary = {'replace this': 'with that',
            >>>              'one': 'two'}
        
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

def backup(executed_modules, path_backup="backup", flag_copy = False, backup_data = False):
    create_folder(path_backup)
    
    modules = ""
    folders = []
    for m in executed_modules:
        modules += "_" + m
        if backup_data is True:
            folders += ["data/" + m]
    
    folders = make_distinct_list(folders)
    
    
    t = time.strftime("%y%m%d_%H%M")
    zip_settings = {'zip_file_name': "{}/{}{}.zip".format(path_backup, t, modules),
                    'zip_include_folder_list': ["config", "lib", "script"] + folders,
                    'zip_include_file_list': ["main.py"],
                    'skipped_folders': [".git", "__pycache__"]}
    
    
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
        header = ["automatically generated numbers"] + header
        for ii in range(len(array)):
            export += [np.array([ii+1, array[ii]])]
    elif len(array_shape) == 2:
        export = array
    elif len(array_shape) > 2:
        print("save_as_csv (max. 2-dimensional array): wrong shape {}".format(np.shape(array_shape)))
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

def csv_to_plot(csv_file_path_list, plot_save_path,
                plot_settings,
                x_column = 0, y_column = [1],
                label = [], label_box_anchor = []):
    """reads multiple csv files and plots specific columns.
    Arguments
    ----
        csv_file_path_list: list<string>
            list of csv file paths
        plot_save_path: string
            plot file path
        plot_settings: dictionary
        
    Plot settings
    -----
        Example plot settings:
            >>> plot_settings = {'suptitle': 'shift',
            >>>                  'xlabel': 'distance / m',
            >>>                  'xmul': 1,
            >>>                  'ylabel': 'calculated shift / um',
            >>>                  'ymul': 1000,
            >>>                  'delimiter': ',',
            >>>                  'skip_rows': 1,
            >>>                  'log x': False,  # optional
            >>>                  'log y': False}  # optional
    """
    p = plt.figure()
    
    for f in csv_file_path_list:
        a = np.loadtxt(f, delimiter=plot_settings['delimiter'],
                       skiprows=plot_settings['skip_rows'])
        
        for i in range(len(y_column)):
            if label == []:
                label_buffer = get_file_name(f) + "_c{}".format(y_column[i])
            else:
                label_buffer = label[i]
            x_value = a[:,x_column] * plot_settings['xmul']
            y_value = a[:,y_column[i]] * plot_settings['ymul']
            plt.plot(x_value, y_value, label=label_buffer)
        
        try: 
            if plot_settings['log y'] is True:
                plt.yscale('log')        
        except KeyError:
            pass
        
        try:
            if plot_settings['log x'] is True:
                    plt.xscale('log')
        except KeyError:
            pass
        
        
        plt.xlabel(plot_settings['xlabel'])
        plt.ylabel(plot_settings['ylabel'])
        
        if label_box_anchor == []:
            plt.legend(loc='best')
        else:
            plt.legend(loc='upper center', bbox_to_anchor=label_box_anchor, ncol=2)
        
        plt.suptitle(plot_settings['suptitle'])
        
    p.savefig(plot_save_path, bbox_inches='tight')
    plt.close(p)
    
def create_array_from_columns(columns):
    """
    Creates an array from several 1-dimensional lists.
    
    Arguments
    ----
        columns: list
            list of 1-dimensional lists
        
    Returns
    ----
        array: list
            2-dimensional
    """
    # http://stackoverflow.com/q/3844948/
    def checkEqualIvo(lst):
        return not lst or lst.count(lst[0]) == len(lst)
    
    column_len = []
    for c in columns:
        column_len += [len(c)]
        
    array = []
    if checkEqualIvo(column_len) is True:
        cols = len(columns)
        rows = column_len[0]
        
        array = [[0] * cols for i in range(rows)]
        
        for row in range(rows):
            for col in range(cols):
                array[row][col] = columns[col][row]
    else:
        print("create_array_from_columns: ERROR")
        print("  unequal column lengths")
        
    return array

def split_list_randome(l, percentage=90, r_element = []):
    """
    Arguments
    ----
        l: list
            1-dimensional list
        percentage: float (optinal)
            first proportion of list *l*
        r_element: list<integer> (optional)
            defines which list element should be in *list_1*
    
    Returns
    ----
        list_1: list
            first proportion of list *l* (*percentage*)
        list_2: list
            secon proportion of list *l* (1-*percentage*)
        r_element: integer
            element numbers of the list *l* that are on the list *list_1*.
    """
    if r_element == []:
        size = int(percentage/100.0*len(l))
        r_element = random.sample(range(1, len(l)), size)
        
    list_1 = []
    list_2 = []
    for i in range(len(l)):
        found = False
        for r in r_element:
            if r == i:
                list_1 += [l[i]]
                found = True
                break
        if found is False:
            list_2 += [l[i]]
            
    return list_1, list_2, r_element