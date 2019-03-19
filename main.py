#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:22:54 2018

@author: maxdh
"""

#import time

from lib.toolbox import toolbox

import script.memory_effect as me

#import script.fog_simulation as fs

#import script.neuronal_network_todo as nn

#import script.histogram as hg

#from lib.f2 import f2

# By calling a module, e.g. F2_main(), the folder name is written to this list.
# This folder is located in the "data" folder and contains all data of the
# module. This will be zipped later.
executed_modules = []

#fs.run()

m = me.memory_effect()
executed_modules = m.run()


#path = "data/f2/output/speckle"
#folder = toolbox.get_subfolders(path)
#folder.sort()
#
#for i in range(len(folder)):
#    folder[i] = path + "/" + folder[i]
#
#
#m.evaluate_data(folder)



#f2.sortToFolderByLayer()

#folders = toolbox.get_subfolders()
#
#nn.init()
#nn.nn_main()
#nn.nn_all_layers()


#folder, path, layer = me.f2_main("", 0)
#for i in range(1,10):
#    executed_modules += ["f2"]
#    folder, path, layer = me.f2_main("", i*5, generate_scatter_plate=False)

    
# ---------------------------------------------
# BACKUP AND DEDUPLICATION BY COMPRESSION
# ---------------------------------------------
toolbox.print_program_section_name("BACKUP AND DEDUPLICATION BY COMPRESSION")

toolbox.backup(executed_modules)

executed_modules = []

#toolbox.send_message("main.py finished")
