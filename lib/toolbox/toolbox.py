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

def get_file_path_with_extension(pathFolder, extension):
    """
    Scans a folder for all files with a specific extension.
    # Arguments
        pathFolder: search folder
        extension: accepts a list of file extensions
        
    # Returns
        list with all files in the _pathFolder_ with a specific _extension_
    
    """
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

def get_file_path_with_extension_include_subfolders(pathFolder, extension):
    """
    returns a list with all files in the _pathFolder_ with a specific _extension_
    # Arguments
        pathFolder: 
        extension: 
    """
    
    if pathFolder[len(pathFolder)-1] != "/":
        pathFolder = pathFolder + "/"
    
    filePath = []
    
    for root, dirs, files in os.walk(pathFolder):
        print(root)
        print("dirs: {}".format((len(dirs))))
        print("files: {}\n".format(len(files)))
    
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

def loadImage(sourcePath, destinationPath, invertColor=False, resize=False, xPixel=0, yPixel=0, prefix=""):
    """
    Loads images and saves them in the path _destinationPath_.
    
    # Arguments
        sourcePath:        list of images paths
        destinationPath:   folder in which  the image is to be saved
        invertColors:      inverts the color of the image
        resize:            if this is true, the image will be resized with _xPixel_ and _yPixel_ parameters.
        xPixel:            number of pixels in x direction, if the image is to be resized
        yPixel:            number of pixels in y direction, if the image is to be resized
        prefix:            adds an optional prefix to the filename
        
    # Retruns
        list of image paths in the _destinationPath_
    """
    os.makedirs(destinationPath, 0o777, True)
    
    rv = []
    for ip in sourcePath:
        fileExtension = os.path.splitext(ip)[-1]
        
        im = Image.open(ip)
        
        im = im.convert("RGB")  # 24 bit: required by F2
        
        if (resize == True):
            im = im.resize((xPixel, yPixel))
        
        if (invertColor == True):
            im = ImageOps.invert(im)
        
        base = os.path.basename(ip)
        name = os.path.splitext(base)
        
        if destinationPath[-1] != "/":
            destinationPath += "/"
            
        if prefix == "":
            path = destinationPath + name[0] + fileExtension
        else:
            path = destinationPath + prefix + name[0] + fileExtension
        
        im.save(path)
        rv = rv + [ path ]
    
    return rv

def load_np_images(pathImage, extension=["npy", "bin"]):
    """
    Loads a list of numpy images and returns it.
    
    # Arguments
        pathImage: path list of numpy images
        exteinsion: specifies the file extension
        
    # Returns
        Lsit of numpy arrays that contain the images.
    """
    image = []
    
    for i in pathImage:
        fileExtension = os.path.splitext(i)[-1]
        for ex in extension:
            if ex[0] != ".":
                ex = "." + ex
            if (fileExtension == ex):
                image = image + [np.load(i)]
        
    return image

def print_program_section_name(name):
    """
    Formats and prints the _name_ on stdout.
    
    # Arguments
        name: string to be printed
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
          
def runProcess(process, arg=[""]):
    "stars a process with an argument; return: output, t: time [s]"
    
    start = time.time()

    print("----- Start Subprocess ----")
    print("Process: {}".format(process))
    print("Argument: {}".format(arg))
    sys.stdout.flush()
    p = subprocess.Popen([process] + arg, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=os.getcwd())
    output = p.communicate()
    exitCode = p.returncode

    exitCode = p.wait()
    print("ExitCode: {}".format(exitCode))
    sys.stdout.flush()
    t = time.time() - start
    
    outputDecoded = []
    for i in range(len(output)-1):
        outputDecoded = outputDecoded + [output[i].decode("utf-8")]
    
    return outputDecoded, exitCode, t