#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:40:35 2019

@author: itodaiber
"""

from lib.toolbox import toolbox

class module_base(object):
    """
    This class creats the folder structure for mudules. This folder structure
    represents the data structure with _input_, _intermediate_data_ and
    _output_ folders.
    
    Example of a derived class
    ----
    
    
    >>> from lib.module_base.module_base import module_base
    >>> 
    >>> class test_module(module_base):
    >>>     def __init__(self, **kwds):
    >>>        super(test_module, self).__init__(name="test_module", **kwds)
    
    """
    def __init__(self, name, **kwds):
        """During initialisation folders and subfolders are created.

        Arguments
        ----
            name : str
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
            path : str
                Path of the data to be copied. [string, list of string]
            subfolder : str
                optional: name of the subfolder in the input directory
            *ignore_pattern: [, ".png", ...]
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

    def load_input_from_module(self, module, subfolder = "", *ignore_pattern):
        """ Copies files and folders in the output folder to the input
        directory.
        
        Arguments
        ----
            module: object derived from *module_base*
                contains the output directory path
            subfolder : str
                optional: name of the subfolder in the input directory
            *ignore_pattern: [, ".png", ...]
                optional: pattern of files to be ignored during copy operation        
        """
        
        path = module.path_output
        
        self.load_input(path, subfolder=subfolder, ignore_pattern)
        
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