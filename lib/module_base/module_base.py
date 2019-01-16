#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:40:35 2019

@author: itodaiber
"""

import os

from lib.toolbox import toolbox

class module_base(object):
    def __init__(self, name, **kwds):
        """Creats folders and subfolders

        Arguments
        ----
            name
                name of the module
        """
        self._module_name = name
        
        self._path_data = "data"
        self._path_module_data = "/{}".format(self._module_name)

        self._path_input = self._path_data + self._path_module_data + "/input"
        self._path_intermediate_data = self._path_data + self._path_module_data + "/intermediate_data"
        self._path_output = self._path_data + self._path_module_data + "/output"
        
        self._path_documentation = self._path_output + "/documentation"
        
        

        toolbox.create_folder(self._path_input)
        toolbox.create_folder(self._path_output)
        toolbox.create_folder(self._path_intermediate_data)
        toolbox.create_folder(self._path_documentation)

    def load_input(self, path, subfolder = "", *ignore_pattern):
        """ Copies files and folders to the input directory.
        
        Arguments
        ----
            path
                Path of the data to be copied. [string, list of string]
            subfolder
                optional: name of the subfolder in the input directory
            *ignore_pattern
                optional: pattern of files to be ignored during copy operation
        """

        if subfolder != "":
            if subfolder[0] != "/":
                subfolder = "/" + subfolder
            self.__create_folder(self.path_input + subfolder)
            
        if type(path) is list:
            for p in path:
                toolbox.copy(path, self.path_input + subfolder, ignore_pattern)
        elif type(path) is str:
            toolbox.copy(path, self.path_input + subfolder, ignore_pattern)

    @property
    def module_name(self):
        """name of the module"""
        return self._module_name
    
    @property
    def path_data(self):
        """relative path of the _data_ directory"""
        return self._path_data
    
    @property
    def path_input(self):
        """relative path of the _input_ directory"""
        return self._path_input
    
    @property
    def path_intermediate_data(self):
        """relative path of the _path_intermediate_data_ directory"""
        return self._path_intermediate_data
    
    @property
    def path_ouput(self):
        """relative path of the _output_ directory"""
        return self._path_ouput

    @property
    def path_documentation(self):
        """relative path of the _documentation_ directory"""
        return self._path_documentation