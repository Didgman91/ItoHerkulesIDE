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
# settings
# ---------------------------------------------

project_name = "hello"

# zip settings
zip_settings = {'zip_File_Name': "herkules.zip",
                'zip_Include_Folder_List': ["config", "lib", "script"],
                'zip_Include_File_List': ["main.py"],
                'skipped_Folders': [".git", "__pycache__"]}

# environment

class env:
    herkules = "herkules"
    herkules_2 = "herkules2"
    bartosz = "bartosz"
    jia = "jia"
    sergej = "sergej"
    zhou = "zhou"
    
environment = env.sergej

#remote_command = "python3.4 main.py"
remote_command = "echo hello grid"

# ---------------------------------------------
# ~settings
# ---------------------------------------------


def get_sftp_settings(project_name, environment, remote_command):
    # sftp settings

    # login without password:
    # - generate an OpenSSH key
    # - save as ~/.ssh/herkules
    # - add the public key to the .ssh/authorized_keys file on the server
    #
    # __Hint__
    #
    # PuTTY
    #   Conversions -> Export OpenSSH key
    #

    remote_user = "itodaiber"
    
    user_Dir = os.path.expanduser("~")
    
    sftp_settings = []
    
    # additional execute after compression and copying
    # For example:
    # execute = "bsub -J {} 'python3.4 main.py'".format(project_name)
    execute = []
    if environment == env.herkules:
        key_file_path = os.path.normpath("{}/.ssh/herkules".format(user_Dir))
        sftp_settings = {'host': "herkules.ito.uni-stuttgart.de",
                         'port': 22,
                         'user_Name': remote_user,
                         'key_File': key_file_path,
                         'path_Remote': "/home/Freenas/{}/{}/".format(remote_user, project_name),
                         'file': zip_settings['zip_File_Name']}
        execute = "bsub -J {} '{}'".format(project_name, remote_command)
    else:
        key_file_path = os.path.normpath("{}/.ssh/herkules2".format(user_Dir))
        sftp_settings = {'host': "herkules2.ito.uni-stuttgart.de",
                         'port': 22,
                         'user_Name': remote_user,
                         'key_File': key_file_path,
                         'path_Remote': "/home/Grid/{}/{}/".format(remote_user, project_name),
                         'file': zip_settings['zip_File_Name']}
        execute = "bsub -R\"select[hname={}]\" -J {} '{}'".format(environment, project_name, remote_command)
    
    
    
    
    return sftp_settings, execute


def zip_Project(zip_Settings):
    """ Zips certain files and folders into a zip file.

    Arguments
    ----
        zip_Settings
            Contains all settings related the compression process. For example
            _zip_File_Name_, a list of files and folders to be zipped and a
            list of files or folders that shouldn't be zipped.
    """
    zip_Include_Folder_List = zip_Settings['zip_Include_Folder_List']
    zip_Include_File_List = zip_Settings['zip_Include_File_List']
    zip_File_Name = zip_Settings['zip_File_Name']

    skipped_Folders = zip_Settings['skipped_Folders']

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
            for f in skipped_Folders:
                if root.find(f) != -1:
                    skip = True
            if skip is False:
                for file in files:
                    zip_h.write(os.path.join(root, file),
                                os.path.relpath(os.path.join(root, file),
                                os.path.join(path, '..')))

    def zip_File(file, zip_h):
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
            zip_File(f, zip_f)
        zip_f.close()

    zip_it(zip_Include_Folder_List, zip_Include_File_List, zip_File_Name)


def copy_To_Server(sftp_settings, execute=""):
    """Copies a zip file via sftp to a folder on a server.

    Arguments
    ----
        sftp_settings
            Contains all settings related the copy process to a server.
            For example host, port and user name.

        execute
            Command to be called after copying and decompression the file.
    """
    host = sftp_settings['host']
    port = sftp_settings['port']

    user_Name = sftp_settings['user_Name']
    key_File = sftp_settings['key_File']

    path_Remote = sftp_settings['path_Remote']

    file_Name = sftp_settings['file']

    # open connection
    ssh_client = paramiko.SSHClient()
    # trust all hosts
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=host,
                       port=port,
                       username=user_Name,
                       key_filename=key_File)

    # creat the directory
    command = "mkdir {}".format(path_Remote)
    stdin, stdout, stderr = ssh_client.exec_command(command)

    for l in stdout.readlines():
        print(l[:-1])

    # transfer the ziped project
    ftp_client = ssh_client.open_sftp()
    ftp_client.put(file_Name, "{}{}".format(path_Remote, file_Name))
    ftp_client.close()

    # unzip project
    command = "cd {pathRemote}\
               && unzip -q -o '{zipFileName}'\
               && rm '{zipFileName}'"

    if execute != "":
        command += "&& {additionalExecute}"

    command = command.format(pathRemote=path_Remote,
                             zipFileName=file_Name,
                             additionalExecute=execute)

    stdin, stdout, stderr = ssh_client.exec_command(command)

    for l in stdout.readlines():
        print(l[:-1])

    ssh_client.close()


# ---------------------------------------------
# Zip the project
# ---------------------------------------------
zip_Project(zip_settings)

# ---------------------------------------------
# Copy project to the server
# ---------------------------------------------
sftp_settings, execute = get_sftp_settings(project_name, environment, remote_command)

copy_To_Server(sftp_settings, execute)

# ---------------------------------------------
# clean up local folder
# ---------------------------------------------
os.remove(zip_settings['zip_File_Name'])
