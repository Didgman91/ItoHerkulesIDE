#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:16:08 2018

@author: daiberma
"""

import os
import zipfile

# ---------------------------------------------
# settings
# ---------------------------------------------

# zip settings
#zip_settings = {'zip_file_name': "herkules.zip",
#                'zip_include_folder_list': ["config", "lib"],
#                'zip_include_file_list': ["main.py"],
#                'skipped_folders': [".git", "__pycache__"]}

# ---------------------------------------------
# ~settings
# ---------------------------------------------


def zip_data(zip_settings):
    """ Zips certain files and folders into a zip file.

    Arguments
    ----
        zip_settings
            Contains all settings related the zipping process. For example
            _zip_file_name_, a list of files and folders to be zipped and a
            list of files or folders that shouldn't be zipped.
    """
    zip_include_folder_list = zip_settings['zip_include_folder_list']
    zip_include_file_list = zip_settings['zip_include_file_list']
    zip_file_name = zip_settings['zip_file_name']

    skipped_folders = zip_settings['skipped_folders']

    def zip_dir(path, zip_h):
        """zip all folder in path, exept those mentioned in _skippedFolders_

        Arguments
        ----
            path
                folder path to zip
            zip_h
                zip handle
        """
        if path[-1] == "/":
            path = path[:-1]
        buffer = os.path.split(path)
        rel = ""
        for b in buffer:
            if b != '':
                rel += "../"
        for root, dirs, files in os.walk(path):
            skip = False
            for f in skipped_folders:
                if root.find(f) != -1:
                    skip = True
            if skip is False:
                for file in files:
                    zip_h.write(os.path.join(root, file),
                                os.path.relpath(os.path.join(root, file),
                                                os.path.join(path, rel)))

    def zip_file(file, zip_h):
        """zip file

        Arguments
        ----
            file
                file to zip
            zip_h
                zip handle
        """
        zip_h.write(file, file)

    def zip_it(dir_list, file_list, zip_name):
        """zip specific folders and files

        Arguments
        ----
            dir_list
                list of folders to zip
            file_lsit
                list of files to zip
            zip_name
                name of the zip file
        """
        zip_f = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
        for dir in dir_list:
            zip_dir(dir, zip_f)
        for f in file_list:
            zip_file(f, zip_f)
        zip_f.close()

    zip_it(zip_include_folder_list, zip_include_file_list, zip_file_name)
