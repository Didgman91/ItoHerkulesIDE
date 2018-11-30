#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:54:57 2018

@author: daiberma
"""

import os
import zipfile

import paramiko

# ---------------------------------------------
# Zip the HerkulesIDE project
# ---------------------------------------------
pathRemote="/home/itodaiber/R - Repositories/tmp/"
zipFileName="herkules.zip"

zipFolder=["config", "lib"]
zipFile=["main.py"]

skippedFolders=[".git", "__pycache__"]

def zipdir(path, ziph):
    """zip all folder in path, exept those mentioned in _skippedFolders_
    
    # Arguments
        path
            folder path to zip
        ziph
            zip handle
    """
    for root, dirs, files in os.walk(path):
        skip = False
        for f in skippedFolders:
            if root.find(f)!=-1:
                skip = True
        if skip==False:
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))

def zip_File(file, ziph):
    """zip file
    
    # Arguments
        file
            file to zip
        ziph
            zip handle
    """
    ziph.write(file, file)

def zipit(dir_list, file_list, zip_name):
    """zip specific folders and files
    
    # Arguments
        dir_list
            list of folders to zip
        file_lsit
            list of files to zip
        zip_name
            name of the zip file
    """
    zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dir in dir_list:
        zipdir(dir, zipf)
    for f in file_list:
        zip_File(f, zipf)
    zipf.close()

zipit(zipFolder, zipFile, zipFileName)


# ---------------------------------------------
# Copy project to the server Herkules
# ---------------------------------------------
paramiko.util.log_to_file('/tmp/paramiko.log')

# Open a transport

host = "herkules.ito.uni-stuttgart.de"
port = 22
transport = paramiko.Transport((host, port))

# Auth
username = "itodaiber"
transport.auth_publickey(username=username, key="~/.ssh/herkules")
transport.connect(username = username)


# Go!

sftp = paramiko.SFTPClient.from_transport(transport)

## Download
#
#filepath = '/etc/passwd'
#localpath = '/home/remotepasswd'
#sftp.get(filepath, localpath)

# Upload

filepath = "home/{}/tmp2/{}".format(username, zipFileName)
localpath = zipFileName
sftp.put(localpath, filepath)

# Close

sftp.close()
transport.close()





#
#
#zip -r -q ./$zipFileName config/ lib/ main.py 
#
#scp ."/$zipFileName" herkules:"'$pathRemote'"
#
#ssh herkules "unzip -q '$pathRemote/$zipFileName' -d '$pathRemote'; rm '$pathRemote/$zipFileName'"
#
#rm ./$zipFileName