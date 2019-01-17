#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:22:54 2018

@author: maxdh
"""

import os

import time

from lib.toolbox import toolbox
from lib.toolbox.zipData import zip_data

import script.memory_effect as me

# By calling a module, e.g. F2_main(), the folder name is written to this list.
# This folder is located in the "data" folder and contains all data of the
# module. This will be zipped later.
executed_modules = []


#folder, path, layer = me.f2_main("", 0)
for i in range(1,10):
    executed_modules += ["f2"]
    folder, path, layer = me.f2_main("", i*5, generate_scatter_plate=False)
    
#    # ---------------------------------------------
#    # CALCULATE MEMORY EFFECT
#    # ---------------------------------------------
#    executed_modules += "memory_effect"
#    toolbox.print_program_section_name("CALCULATE MEMORY EFFECT")
#    
#    path_memory_effect = "data/memory_effect/input/variable_fog"
#    os.makedirs(path_memory_effect, 0o777, True)
#    
#    path_speckle = path_memory_effect + "/{},0".format(pow(10,i))
#    
#    os.makedirs(path_speckle, 0o777, True)
#    
#    for f in folder:
#        if f[-1] == "/":
#            f = f[:-1]
#        destination_folder_name = os.path.split(f)[-1]
#        toolbox.copy_folder(f, path_speckle + "/" + destination_folder_name)
    
# ---------------------------------------------
# BACKUP AND DEDUPLICATION BY COMPRESSION
# ---------------------------------------------
toolbox.print_program_section_name("BACKUP AND DEDUPLICATION BY COMPRESSION")

path_backup = "backup"
os.makedirs(path_backup, 0o777, True)

modules = ""
folders = []
for m in executed_modules:
    modules += "_" + m
    folders += ["data/" + m]

folder = toolbox.make_distinct_list(folder)

t = time.strftime("%y%m%d_%H%M")
zip_settings = {'zip_file_name': "{}/{}{}.zip".format(path_backup, t, modules),
                'zip_include_folder_list': ["config", "lib"] + folders,
                'zip_include_file_list': ["main.py"],
                'skipped_folders': [".git", "__pycache__"]}

flag_copy = True

def copy_files(data, destination):
    for f in data:
        toolbox.copy(f, "{}/{}{}".format(path_backup, t, modules))

if flag_copy is True:
    toolbox.create_folder("{}/{}{}".format(path_backup, t, modules))
    copy_files(zip_settings["zip_include_folder_list"] + zip_settings["zip_include_file_list"])
else:
    zip_data(zip_settings)
    

executed_modules = []

#toolbox.send_message("main.py finished")
