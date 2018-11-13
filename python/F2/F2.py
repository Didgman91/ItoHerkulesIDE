#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:07:11 2018

@author: maxdh
"""

import subprocess
import time

def run_script(path):
    "starts the F2 programm with the path of the script file as argument"

    processName = "F2"

    output, exitCode, t = run_Process(processName, path)
    
    return output, exitCode, t

def run_Process(process, arg=""):
    "stars a process with an argument; return: output, t: time [s]"
    
    start = time.time()
    
    process = subprocess.Popen([process, arg], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = process.communicate()
    exitCode = process.returncode

    outputDecoded = {}
    for i in range(len(output)-1):
        outputDecoded[i] = output[i].decode("utf-8")
    
    t = time.time() - start
    
    return outputDecoded, exitCode, t
