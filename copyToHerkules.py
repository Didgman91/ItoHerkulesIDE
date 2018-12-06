#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:54:57 2018

@author: daiberma
"""

import os

import zipfile
import paramiko




# zip settings
zip_Settings = {'zip_File_Name': "herkules.zip",
                'zip_Include_Folder_List': ["config", "lib"],
                'zip_Include_File_List': ["main.py"],
                'skipped_Folders': [".git", "__pycache__"]}

# sftp settings
user_Dir = os.path.expanduser("~")
sftp_Settings = {'host': "herkules.ito.uni-stuttgart.de",
                 'port': 22,
                 'user_Name': "itodaiber",
                 'key_File': "{}/.ssh/herkules".format(user_Dir),
                 'path_Remote': "./tmp/",
                 'file': zip_Settings['zip_File_Name']}



def zip_Project(zip_Settings):
    """ Zips certain files and folders into a zip file.
    
    # Arguments
        zip_Settings
            Contains all settings related the zipping process. For example
            _zip_File_Name_, a list of files and folders to be zipped and a
            list of files or folders that shouldn't be zipped.
    """
    zip_Include_Folder_List = zip_Settings['zip_Include_Folder_List']
    zip_Include_File_List = zip_Settings['zip_Include_File_List']
    zip_File_Name = zip_Settings['zip_File_Name']
    
    skipped_Folders = zip_Settings['skipped_Folders']
    
    def zip_dir(path, zip_h):
        """zip all folder in path, exept those mentioned in _skippedFolders_
        
        # Arguments
            path
                folder path to zip
            zip_h
                zip handle
        """
        for root, dirs, files in os.walk(path):
            skip = False
            for f in skipped_Folders:
                if root.find(f)!=-1:
                    skip = True
            if skip==False:
                for file in files:
                    zip_h.write(os.path.join(root, file),
                               os.path.relpath(os.path.join(root, file),
                                               os.path.join(path, '..')))
    
    def zip_File(file, zip_h):
        """zip file
        
        # Arguments
            file
                file to zip
            zip_h
                zip handle
        """
        zip_h.write(file, file)
    
    def zip_it(dir_list, file_list, zip_name):
        """zip specific folders and files
        
        # Arguments
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
            zip_File(f, zip_f)
        zip_f.close()
    
    zip_it(zip_Include_Folder_List, zip_Include_File_List, zip_File_Name)



def copy_To_Server(sftp_Settings, execute=""):
    """Copies a zip file via sftp to a folder on a server.
    
    # Arguments
        sftp_Settings
            Contains all settings related the copy process to a server.
            For example host, port and user name.
        
        execute
            Command to be called after copying and unzipping the file.
    """
    host = sftp_Settings['host']
    port = sftp_Settings['port']
    
    user_Name = sftp_Settings['user_Name']
#    key_File="/home/daiberma/.ssh/herkules"
    key_File = sftp_Settings['key_File']
    
    path_Remote = sftp_Settings['path_Remote']
    
    file_Name = sftp_Settings['file']
    
    # open connection
    ssh_client=paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # trust all hosts
    ssh_client.connect(hostname=host,
                       port=port,
                       username=user_Name,
                       key_filename=key_File)
    
    
    # creat the directory
    stdin,stdout,stderr=ssh_client.exec_command("mkdir {}".format(path_Remote))
    #stdin,stdout,stderr=ssh_client.exec_command("cd ./tmp && ls")
    #stdin,stdout,stderr=ssh_client.exec_command("ls")
    
    for l in stdout.readlines():
        print(l[:-1])
    
    
    # transfer the ziped project
    ftp_client=ssh_client.open_sftp()
    ftp_client.put(file_Name, "{}{}".format(path_Remote, file_Name))
    ftp_client.close()
    
    
    # unzip project
    command = "cd {pathRemote}\
               && unzip -q -o '{zipFileName}'\
               && rm '{zipFileName}'\
               && {additionalExecute}"
                    
    command = command.format(pathRemote=path_Remote,
                             zipFileName=file_Name,
                             additionalExecute=execute)
    
    stdin,stdout,stderr = ssh_client.exec_command(command)
    
    for l in stdout.readlines():
        print(l[:-1])
    
    ssh_client.close()


# ---------------------------------------------
# Zip the HerkulesIDE project
# ---------------------------------------------
zip_Project(zip_Settings)

# ---------------------------------------------
# Copy project to the server Herkules
# ---------------------------------------------
copy_To_Server(sftp_Settings)
#copy_To_Server(sftp_Settings, "bsub 'python3.4 main.py'")

# ---------------------------------------------
# clean up local folder
# ---------------------------------------------
os.remove(zip_Settings['zip_File_Name'])
