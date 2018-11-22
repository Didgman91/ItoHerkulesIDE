#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:07:11 2018

@author: maxdh
"""
import os

from PIL import Image
from PIL import ImageOps

from ..toolbox.toolbox import runProcess

#from ..DataIO import mnistLib as mnist

pathData = "data"

pathScript = pathData + "/F2/intermediateData/script"
pathScatterPlate = pathData + "/F2/intermediateData/scatterPlate"

pathInput = "/F2/input"
pathInputNIST = "/F2/input/NIST"
pathIntermediateDataScatterPlate = "/F2/intermediateData/scatterPlate"
pathIntermediateDataScript = "/F2/intermediateData/script"
pathOutputSpeckle = "/F2/output/speckle"
pathOutputDocumentation = "/F2/output/documentation"

fileNameScriptCreateScatterPlate = "createScatterPlate.txt"
fileNameScatterPlateRandom = ["ScatterPlateRandomX", "ScatterPlateRandomY"]

def run_script(path):
    "starts the F2 programm with the path of the script file as argument"

    processName = "F2"

    output, exitCode, t = runProcess(processName, [path])
    
    return output, exitCode, t

def generate_folder_structure(path="data"):
    "Creats folders and subfolders related to the F2 process in the folder _path_."
    os.makedirs(path+pathInput, 0o777, True)
    os.makedirs(path+pathIntermediateDataScatterPlate, 0o777, True)
    os.makedirs(path+pathIntermediateDataScript, 0o777, True)
    os.makedirs(path+pathOutputSpeckle, 0o777, True)
    os.makedirs(path+pathOutputDocumentation, 0o777, True)
    
def create_scatter_plate(parameter=""):
    "Creats the scatterplate and saves it in the folder _path_."
    
    if parameter == "":
        parameter = get_F2_script_parameter()
    
    textFile = open("config/F2/ScriptPartCreateScatterPlate.txt", "r")
    lines = textFile.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i].format(py_pathScatterPlateX=pathScatterPlate + "/" +fileNameScatterPlateRandom[0], py_pathScatterPlateY=pathScatterPlate + "/" +fileNameScatterPlateRandom[1])
    
    
    script = parameter + lines
    
    with open(pathScript + "/" + fileNameScriptCreateScatterPlate, "w") as text_file:
        for i in range(len(script)):
            print(script[i], file=text_file)
    
    run_script(pathScript + "/" + fileNameScriptCreateScatterPlate)
    
    scatterPlateRandom = [pathScatterPlate + "/" +fileNameScatterPlateRandom[0],
                          pathScatterPlate + "/" +fileNameScatterPlateRandom[1]]
    
    return scatterPlateRandom
            
def calculate_propagation(imagePath, scatterPlateRandom):
    
    parameterScript = get_F2_script_parameter()
    
    imageScript = []
    propagateScript = []
    for i in range(len(imagePath)):
        rv = get_F2_script_load_image(imagePath[i])
        imageScript = imageScript + [rv]
        
        rv = get_F2_script_propagete(imagePath[i], scatterPlateRandom)
        propagateScript = propagateScript + [rv]
    
    
    for i in range(len(imagePath)):
        script = parameterScript + imageScript[i] + propagateScript[i]
        
        with open(pathScript + "/calculatePropagation.txt", "w") as text_file:
            for ii in range(len(script)):
                print(script[ii], file=text_file)
        
        print("F2 propagation calculation: Image {}/{}".format(i, len(imagePath)))
        run_script(pathScript + "/calculatePropagation.txt")

def get_F2_script_parameter():
    
    textFile = open("config/F2/ScriptPartSetParameters.txt", "r")
    lines = textFile.readlines()
    
    return lines

#def load_MNIST_train_images(pathMNIST, imageNumbers):
#    os.makedirs(pathData+"/F2/input/MNIST", 0o777, True)
#    
#    images, lables = mnist.load_train_data(pathMNIST)
#    
#    imagePath = []
#    
#    for i in range(len(imageNumbers)):
#        imagePath = imagePath + [pathData+"/F2/input/MNIST/image{:06}.bmp".format(imageNumbers[i])]
#        mnist.save_as_bmp(images[imageNumbers[i]],
#                          pathData+"/F2/input/MNIST/image{:06}.bmp".format(imageNumbers[i]))
#        
#    return imagePath

def load_image(imagePath, invertColor=False, resize=False, xPixel=0, yPixel=0):
    "loads the frist _imageNumbers_ images from _pathNIST_"
    os.makedirs(pathData + pathInputNIST, 0o777, True)
    
    rv = []
    for ip in imagePath:
        im = Image.open(ip)
        
        im = im.convert("RGB")  # 24 bit: required by F2
        
        if (resize == True):
            im = im.resize((xPixel, yPixel))
        
        if (invertColor == True):
            im = ImageOps.invert(im)
        
        base = os.path.basename(ip)
        name = os.path.splitext(base)
        
        path = pathData + pathInputNIST + "/" + name[0] + ".bmp"
        im.save(path)
        rv = rv + [ path ]
    
    return rv
    

def get_F2_script_load_image(file):
    ""
    textFile = open("config/F2/ScriptPartLoadImage.txt", "r")
    lines = textFile.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i].format(py_fileName=file)
    
    return lines

def get_F2_script_propagete(fileName, scatterPlateRandom):
    "returns only the part of the script to calculate the electrical field"
    outputPath = pathData + "/F2/output/speckle"
    
    fileName = os.path.basename(fileName)
    fileName = os.path.splitext(fileName)[0]
    
    textFile = open("config/F2/ScriptPartCalculatePropagation.txt", "r")
    lines = textFile.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i].format(py_scatterPlateRandomX=scatterPlateRandom[0], py_scatterPlateRandomY=scatterPlateRandom[1], py_outputPath=outputPath, py_fileName=fileName)
            
    return lines