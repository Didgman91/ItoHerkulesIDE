#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:07:11 2018

@author: maxdh
"""

import subprocess
import time
import os

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
    process = subprocess.Popen([process, arg], cwd=os.getcwd())#, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=os.getcwd())
#    output = process.communicate()
#    exitCode = process.returncode

    exitCode = process.wait()
    t = time.time() - start
    
    outputDecoded = {}
#    for i in range(len(output)-1):
#        outputDecoded[i] = output[i].decode("utf-8")
    
    return outputDecoded, exitCode, t

def generate_folder_structure(path="data"):
    "Creats folders and subfolders related to the F2 process in the folder _path_."
    os.makedirs(path+"/F2/input/objects", 0o777, True)
    os.makedirs(path+"/F2/intermediateData/scatterPlate", 0o777, True)
    os.makedirs(path+"/F2/intermediateData/script", 0o777, True)
    os.makedirs(path+"/F2/output/speckle", 0o777, True)
    os.makedirs(path+"/F2/output/documentation", 0o777, True)
    
def create_scatterplate(parameter):
    "Creats the scatterplate and saves it in the folder _path_."
    
    pathScript = "data/F2/intermediateData/script/createScatterPlate.txt"
    
    pathScatterPlate = "data/F2/intermediateData/scatterPlate"
    
    text = ["x=RandomReal p1,p2,streuanz",
            "y=RandomReal p3,p4,streuanz ",
            "Save \"{}\", x".format(pathScatterPlate + "/ScatterPlateX"),
            "Save \"{}\", y".format(pathScatterPlate + "/ScatterPlateY")]
    
    script = parameter + text
    
    with open(pathScript, "w") as text_file:
        for i in range(len(script)):
            print(script[i], file=text_file)
    
    run_script(pathScript)
            

def get_Parameter():
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
            "rhon=0.2 !g/m**3",
            "dist=10*m  ",
            "dp=0.1*m    ! Simulationsgebiet",
            "sam=4096 !*3      ! Sampling",
            "max=1       ! Anzahl Schichten",
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

def load_image():
    text = ["BMPInit iPixelX,iPixelY ",
            "BMPLoad \"Ampelmann1.bmp\" ! 24Bit RGB",
            "BMP2Array h1",
            "h2=h1!MirrorX h1 ",
            "obj=Zeros[sam,sam]",
            "z=sam/2",
            "v=2*4/2",
            "MatrixInsert obj,h2, z-v*iPixelX,z-v*iPixelY,z+v*iPixelX,z+v*iPixelY, Substitute",
            "NormalizeMax obj",
            "Clear h1,h2",
            "!****************************"]
    
    return text