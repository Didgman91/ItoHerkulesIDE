#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:07:11 2018

@author: maxdh
"""

import subprocess
import time
import os
import sys

from PIL import Image
from PIL import ImageOps

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

    output, exitCode, t = run_process(processName, path)
    
    return output, exitCode, t

def run_process(process, arg=""):
    "stars a process with an argument; return: output, t: time [s]"
    
    start = time.time()

    print("----- Start Subprocess ----")
    print("Process: {}".format(process))
    print("Argument: {}".format(arg))
    sys.stdout.flush()
    process = subprocess.Popen([process, arg], cwd=os.getcwd())#, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=os.getcwd())
#    output = process.communicate()
#    exitCode = process.returncode

    exitCode = process.wait()
    print("ExitCode: {}".format(exitCode))
    sys.stdout.flush()
    t = time.time() - start
    
    outputDecoded = {}
#    for i in range(len(output)-1):
#        outputDecoded[i] = output[i].decode("utf-8")
    
    return outputDecoded, exitCode, t

def generate_folder_structure(path="data"):
    "Creats folders and subfolders related to the F2 process in the folder _path_."
    os.makedirs(path+pathInput, 0o777, True)
    os.makedirs(path+pathIntermediateDataScatterPlate, 0o777, True)
    os.makedirs(path+pathIntermediateDataScript, 0o777, True)
    os.makedirs(path+pathOutputSpeckle, 0o777, True)
    os.makedirs(path+pathOutputDocumentation, 0o777, True)
    
def create_scatter_plate(parameter):
    "Creats the scatterplate and saves it in the folder _path_."
        
    text = ["x=RandomReal p1,p2,streuanz",
            "y=RandomReal p3,p4,streuanz ",
            "Save \"{}\", x".format(pathScatterPlate + "/" +fileNameScatterPlateRandom[0]),
            "Save \"{}\", y".format(pathScatterPlate + "/" +fileNameScatterPlateRandom[1])]
    
    script = parameter + text
    
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
            for i in range(len(script)):
                print(script[i], file=text_file)
        
        run_script(pathScript + "/calculatePropagation.txt")

def get_F2_script_parameter():
    text = ["! *************************************************************",
            "! Beugung an Nebel-Partikel (homogene groesse)",
            "! Bestimmung totale, kollimierte und diffuse Reflexion ",
            "! in Abhaengigkeit der Propagationstiefe",
            "! *************************************************************",
            "",
            "!Dnebel=10-20 mu",
            "!rho_nebel=0.01 - 0.3 g /m^3",
            "",
            "mu=1000",
            "mm=1000*mu",
            "m=1000*mm",
            "rho=997*1000 !dichte Wasser g/m**3",
            "nwasser=1.33",
            "",
            "!**************************** Nebelparameter",
            "d=40*mu",
            "rhon=0.3 !g/m**3",
            "dist=10*m  ",
            "dp=0.1*m    ! Simulationsgebiet",
            "sam=4096 !*3      ! Sampling",
            "max=2       ! Anzahl Schichten",
            "!****************************",
            "",
            "",
            "!**************************** Beleuchtungsparameter/Objekt",
            "lam=514",
            "iPixelX=64!640",
            "iPixelY=iPixelX!400",
            "",
            "",
            "!**************************** Exportparameter",
            "ePixelX=64",
            "ePixelY=ePixelX",
            "!****************************",
            "",
            "? \"Durchmesser Nebeltropfen [nm]         : \",d",
            "? \"Durchmesser Nebeltropfen in Pixel     : \",d/dp*sam",
            "v=4/3*Pi[]*(d/2/m)**3",
            "? \"Volumen Nebeltropfen [m**3]           : \",v",
            "",
            "mnebel=v*rho  ! masse wasser pro nebeltropfen",
            "? \"Masse Nebeltropfen [g]                : \",mnebel",
            "",
            "nm3=rhon/mnebel ! Anzahl Nebeltropfen / m**3",
            "? \"Anzahl Nebeltropfen / m**3            : \",nm3",
            "? \"Anzahl Nebeltropfen / mm Schichtdicke : \",nm3/1000",
            "",
            "p1=-dp/2 ! Simulationsgebiet",
            "p2=dp/2",
            "p3=-dp/2",
            "p4=dp/2",
            "",
            "x0=dist/max    ! Schichtdicke ",
            "? \"Schichtdicke [m]                      : \",x0/m",
            "",
            "streuanz=nm3*(x0*dp*dp/(m**3))",
            "? \"Anzahl Nebeltropfen Schicht           : \",streuanz"]
    
    return text

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

def load_NIST_image(imagePath, invertColor=False, resize=False, xPixel=0, yPixel=0):
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
    text = ["BMPInit iPixelX,iPixelY ",
            "BMPLoad \"{}\" ! 24Bit RGB".format(file),
            "BMP2Array h1",
            "h2=h1!MirrorX h1 ",
            "obj=Zeros[sam,sam]",
            "z=sam/2",
            "v=40",
            "MatrixInsert obj,h2, z-v*iPixelX,z-v*iPixelY,z+v*iPixelX,z+v*iPixelY, Substitute",
            "NormalizeMax obj",
            "Clear h1,h2",
            "!****************************"]
    
    return text

def get_F2_script_propagete(fileName, scatterPlateRandom):
    "returns only the part of the script to calculate the electrical field"
    outputPath= pathData + "/F2/output/speckle"
    
    fileName = os.path.basename(fileName)
    fileName = os.path.splitext(fileName)[0]
    
    text = ["? \" \"",
            "? \"IFeld     \",\"Ip        \",\"Tk        \",\"Td        \",\"Ttot\"",
            "? \"--------------------------------------------------------------------\"",
            "? \" \"",
            "",
            "winkel=0",
            "",
            "ac(1:sam,1:sam)=Cmplx[1,0]",
            "erg1(1:max)=0.0",
            "erg2(1:max)=0.0",
            "erg3(1:max)=0.0",
            "erg4(1:max)=0.0",
            "erg5(1:max)=0.0",
            "",
            "Do j,1,max",
            " ",
            " ax(1:sam,1:sam)=Cmplx[1,0]",
            " Grid ax,p1,p2,p3,p4",
            " ",
            "! x=RandomReal p1,p2,streuanz ",
            "! y=RandomReal p3,p4,streuanz ",
            " x=Load \"{}\"".format(scatterPlateRandom[0]),
            " y=Load \"{}\"".format(scatterPlateRandom[1]),
            " Sphere ax,x,y,d/2,d/2,nwasser,0,lam",
            "",
            " If j .EQ. 1 ! gilt nur fuer erste Schicht",
            "   ax(1:sam,1:sam)=Cmplx[1,0]",
            "   PupilFilter ax,obj ",
            "   Grid ax,p1,p2,p3,p4  ",
            "   a2=Illumination ax,PlaneWave, 1,0, 0,0,lam ! Beleuchtung mit Planwelle ",
            "   !a2=Illumination ax,Gauss, 1,0,  10*mm , 10*mm,0,0, lam ! (x,y,z sind shift-Werte) ! Beleuchtung mit Gauss",
            "   !a2=Illumination ax,AGauss, 1,0,  10*mm,20*mm , 10*mm,0,0, lam ! (x,y,z sind shift-Werte) ! Beleuchtung mit Gauss",
            "   !PlotArray Abs[a2],\"Objekt\",400,400",
            " Else",
            "   PupilFilter a2,ax ! ab zweiter Schicht, belegt Array mit Filter",
            " EndIf",
            "",
            " PwPropagationNF a2,lam,dp,dp,1,x0 ! Propagiert Feld a2 mit FFT-Beampropagation",
            " ",
            "",
            "! If MOD[j,5] .eq. 1",
            "! If j .EQ. 1",
            "   feld=ArrayResize2D Abs[a2], ePixelX,ePixelY",
            "      ",
            "   intensity=Intensity feld   ",
            "",
            "   !feldMX=MirrorX intensity ",
            "   !feldMY=MirrorY intensity  ",
            "   !feldMYR90=Rotate90 feldMY",
            "",
            "   BMPInit ePixelX,ePixelY",
            "   BMPSetPen 255,255,255",
            "   BMPSetPen2 128,128,128",
            "   ",
            "   BMPPlot feld, ePixelX, ePixelY",
            "   BMPSave \"{}/Feld_{}_.bmp\"".format(outputPath, fileName),
            "",
            "   BMPClear",
            "",
            "   BMPInit ePixelX,ePixelY",
            "   BMPSetPen 255,255,255",
            "   BMPSetPen2 128,128,128",
            "   ",
            "   BMPPlot intensity, ePixelX, ePixelY",
            "   BMPSave \"{}/Intensitaet_{}.bmp\"".format(outputPath, fileName),
            "",
            "   ",
            "   SaveNPY \"{}/Feld_{}_\", feld, j".format(outputPath, fileName),
            "   SaveNPY \"{}/Intensitaet_{}_\", intensity, j".format(outputPath, fileName),
            "! EndIf",
            " ",
            " a0=EnergyDensity a2",
            " b=Pupil a2,lam,dp,dp,1 ! FourierTransfo",
            " ",
            " c0=IntegralIntensity b  ! Gesamtintensität",
            " d0=Intensity[b(#+1,#+1)]    ! Kollimierte Transmission",
            " b(#+1,#+1)=Cmplx[0]",
            " e0=IntegralIntensity b  ! Diffuse Transmission ",
            " f0=d0+e0                ! Totale Transmission",
            " ",
            " erg1(j)=a0",
            " erg2(j)=c0",
            " erg3(j)=d0/c0",
            " erg4(j)=e0/c0",
            " erg5(j)=f0/c0",
            "",
            " ? '(F0.7)'erg1(j),'(1X,E9.2)'erg2(j),'(2X,F0.7)'erg3(j),'(2X,F0.7)'erg4(j),'(2X,F0.7)'erg5(j)",
            "",
            "EndDo ! j",
            "",
            " ",
            "BMPInit 850,700",
            "BMPSetPen 255,255,255",
            "BMPSetPen2 128,128,128",
            "",
            "BMPColorMap BW",
            "BMPSetWin 10,10,310,300, 1,1,sam,sam",
            "BMPListPlot1d erg1,\"GesamtEnergie Feld\",\"z\",\"Efeld\"",
            "",
            "BMPSetWin 450,10,750,310, 1,1,3,3",
            "BMPListPlot1d erg2,\"Energie Pupille\",\"z\",\"Epup\"",
            "",
            "BMPSetWin 10,360,310,660, 1,1,sam,sam",
            "BMPListPlot1d2 erg3,erg4,\"Koll./Diff. Transmission\",\"z\",\"T\"",
            "",
            "BMPSetWin 450,360,750,660, 1,1,3,3",
            "BMPListPlot1d erg5,\"Transmission\",\"z\",\"T\"",
            "",
            "BMPSave \"{}/nebel_{}_.bmp\"".format(outputPath, fileName),
            "",
            "End "]
            
    return text