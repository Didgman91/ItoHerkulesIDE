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
        fileExtension = os.path.splitext(name)[1]
        for ex in extension:
            if ex[0] != ".":
                ex = "." + ex
            if (fileExtension == ex):
                filePath = filePath + [ pathFolder + name]
            
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