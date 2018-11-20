#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:54:05 2018

@author: maxdh
"""

import os

def get_file_path_with_extension(pathFolder, extension):
    "returns a list with all files in the _pathFolder_ with a specific _extension_"
    if pathFolder[len(pathFolder)-1] != "/":
        pathFolder = pathFolder + "/"
    
    fileName = os.listdir(pathFolder)
    
    filePath = []
    
    for name in fileName:
        name
        fileExtension = os.path.splitext(name)[1]
        for ex in extension:
            if ex[0] != ".":
                ex = "." + ex
            if (fileExtension == ex):
                filePath = filePath + [ pathFolder + name]
            
    return filePath

def get_file_path_with_extension_include_subfolders(pathFolder, extension):
    "returns a list with all files in the _pathFolder_ with a specific _extension_"
    if pathFolder[len(pathFolder)-1] != "/":
        pathFolder = pathFolder + "/"
    
    filePath = []
    
    for root, dirs, files in os.walk(pathFolder):
        print(root)
        print("dirs: {}".format((len(dirs))))
        print("files: {}".format(len(files)))
    
        for name in files:
            fileExtension = os.path.splitext(name)[1]
            for ex in extension:
                if ex[0] != ".":
                    ex = "." + ex
                if (fileExtension == ex):
                    if root[len(root)-1] != "/":
                        rootBuffer = root + "/"
                    filePath = filePath + [ rootBuffer + name]
            
    return filePath


def print_program_section_name(name):
    
    line = ""
    length = 45
    if length < len(name) + 5:
        length = len(name) + 5
    
    for i in range(length):
        line = line + "-"
    
    print("# {}".format(line))
    print("# {}".format(name))
    print("# {}".format(line))