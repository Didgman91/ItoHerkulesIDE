#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:07:11 2018

@author: maxdh
"""
import os

from PIL import Image
from PIL import ImageOps

from ..toolbox import toolbox

#from ..DataIO import mnistLib as mnist

pathData = "data"

pathScript = pathData + "/f2/intermediate_data/script"
pathScatterPlate = pathData + "/f2/intermediate_data/scatter_plate"

pathInput = "/f2/input"
pathInputNIST = "/f2/input/nist"
pathIntermediateDataScatterPlate = "/f2/intermediate_data/scatter_plate"
pathIntermediateDataScript = "/f2/intermediate_data/script"
pathIntermediateDataStdout = "/f2/intermediate_data"
pathOutputSpeckle = "/f2/output/speckle"
pathOutputDocumentation = "/f2/output/documentation"

fileNameScriptCalculatePropagation = "calculate_propagation.txt"
fileNameScriptCreateScatterPlate = "create_scatter_plate.txt"
fileNameScatterPlateRandom = ["scatter_plate_random_x", "scatter_plate_random_y"]
fileNameStdout = "stdout.txt"

def run_script(path, printStdout = True):
    "starts the F2 programm with the path of the script file as argument"

    processName = "F2"

    output, exitCode, t = toolbox.run_process(processName, [path])

    if printStdout:
        for i in output:
            print(i)
    
    with open(pathData+pathIntermediateDataStdout+"/"+fileNameStdout, "a") as stdoutFile:
        for o in output:
            stdoutFile.write(o)
    
    return output, exitCode, t

def generate_folder_structure(path="data"):
    "Creats folders and subfolders related to the F2 process in the folder _path_."
    os.makedirs(path+pathInput, 0o777, True)
    os.makedirs(path+pathIntermediateDataScatterPlate, 0o777, True)
    os.makedirs(path+pathIntermediateDataScript, 0o777, True)
    os.makedirs(path+pathIntermediateDataStdout, 0o777, True)
    os.makedirs(path+pathOutputSpeckle, 0o777, True)
    os.makedirs(path+pathOutputDocumentation, 0o777, True)
    
def create_scatter_plate(numberOfLayers, distance, parameter=""):
    "Creats the scatterplate and saves it in the folder _path_."
    
    if parameter == "":
        parameter = get_f2_script_parameter(numberOfLayers, distance)
    
    textFile = open("config/F2/ScriptPartCreateScatterPlate.txt", "r")
    lines = textFile.readlines()
    
    for i in range(len(lines)):
        pathX = pathScatterPlate + "/" + fileNameScatterPlateRandom[0]
        pathY = pathScatterPlate + "/" + fileNameScatterPlateRandom[1]
        lines[i] = lines[i].format(py_pathScatterPlateX=pathX, py_pathScatterPlateY=pathY)
        
    script = parameter + lines
    
    with open(pathScript + "/" + fileNameScriptCreateScatterPlate, "w") as text_file:
        for i in range(len(script)):
            print(script[i], file=text_file)
    
    run_script(pathScript + "/" + fileNameScriptCreateScatterPlate)
    
    scatterPlateRandom = []
    scatterPlateRandom += [pathScatterPlate + "/" +fileNameScatterPlateRandom[0],
                          pathScatterPlate + "/" +fileNameScatterPlateRandom[1]]
    
    return scatterPlateRandom
            
def calculate_propagation(imagePath, scatterPlateRandom, numberOfLayers, distance):
    
    parameterScript = get_f2_script_parameter(numberOfLayers, distance)
    
    imageScript = []
    propagateScript = []
    for i in range(len(imagePath)):
        rv = get_f2_script_load_image(imagePath[i])
        imageScript = imageScript + [rv]
        
        rv = get_f2_script_propagete(imagePath[i], scatterPlateRandom)
        propagateScript = propagateScript + [rv]
    
    
    for i in range(len(imagePath)):
        script = parameterScript + imageScript[i] + propagateScript[i]
        
        with open(pathScript + "/" + fileNameScriptCalculatePropagation, "w") as text_file:
            for ii in range(len(script)):
                if script[ii][-1] == '\n':
                    script[ii] = script[ii][:-1]
                print(script[ii], file=text_file)
        
        print("F2 propagation calculation: Image {}/{}".format(i+1, len(imagePath)))
        run_script(pathScript + "/" + fileNameScriptCalculatePropagation)
    
    return 

def sortToFolderByLayer(folderPath = pathData + pathOutputSpeckle, keyword="layer"):
    "moves the files with the extension 'layerxxxx' to a corresponding folder 'layerxxxx'"
    rvPath = []
    rvFolder = []
    rvLayerNumber = []
    
    filePath = toolbox.get_file_path_with_extension(folderPath, ["bmp"])
    
    for fp in filePath:
        # create folder (path + keyword + layerNumber)
        base = os.path.basename(fp)
        name = os.path.splitext(base)
        
        startStrLayerNumber = name[0].find(keyword) + len(keyword)
        if startStrLayerNumber == -1:
            continue
        
        layerNumber = name[0][startStrLayerNumber:]
        
        folder = fp[:-len(base)] + keyword + layerNumber
        os.makedirs(folder, 0o777, True)
        
        # move file into the new folder
        path = folder + "/" + base
        os.rename(fp, path)
        
        rvFolder += [folder]
        rvPath += [path]
        rvLayerNumber += [layerNumber]
        
    return rvFolder, rvPath, rvLayerNumber
    
#def load_mNIST_train_images(pathMNIST, imageNumbers):
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
    

def get_f2_script_parameter(numberOfLayers, distance):
    
    textFile = open("config/f2/ScriptPartSetParameters.txt", "r")
    lines = textFile.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i].format(py_numberOfLayers=numberOfLayers,
             py_distance=distance)
    
    return lines

def get_f2_script_load_image(file):
    ""
    textFile = open("config/f2/ScriptPartLoadImage.txt", "r")
    lines = textFile.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i].format(py_fileName=file)
    
    return lines

def get_f2_script_propagete(fileName, scatterPlateRandom):
    "returns only the part of the script to calculate the electrical field"
    outputPath = pathData + pathOutputSpeckle
    
    base = os.path.basename(fileName)
    name = os.path.splitext(base)
    
    textFile = open("config/f2/ScriptPartCalculatePropagation.txt", "r")
    lines = textFile.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i].format(py_scatterPlateRandomX=scatterPlateRandom[0], py_scatterPlateRandomY=scatterPlateRandom[1], py_outputPath=outputPath, py_fileName=name[0])
            
    return lines