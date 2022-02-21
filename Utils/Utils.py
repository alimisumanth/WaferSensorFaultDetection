# -*- coding: utf-8 -*-

"""
=============================================================================
Created on: 08-02-2022 11:23 PM
Created by: ASK
=============================================================================

Project Name: WaferSensorFaultDetection

File Name: Utils.py

Description : This module contains general/common functions used in this project

Version: 1.0

Revision: None

=============================================================================
"""

import os
import json
import joblib
import shutil
import yaml


class utils:
    """
    common methods which are used in this project

    Method:

      dirCheck(path):
        Creates directories recursively
      mdm(config_path):
        reads and returns master data management file
    """

    def __init__(self):
        self.new_model_directory = None
        self.path = None
        self.model_directory = 'models/'
        self.config = ''
        self.configPath = 'params.yaml'

    def dirCheck(self, path):
        """
        Creates directories recursively

        Args
         path: path whose directory is to created

        Returns: None

        Raises:
          OSError: os related error like permission denied
        """
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print('Permission denied', str(e))
        except Exception as e:
            print('Exception occurred', str(e))

    def mdm(self, config_path):
        """
        Reads and return master data management file

        Args
         config_path: path of the master data management file

        Returns: master data management file in dic format

        Raises:
          OSError: Operating system errors
        """
        try:
            with open(config_path) as file:
                self.config = json.load(file)
        except OSError as exception:
            print(exception)
        return self.config

    def loadYaml(self):
        """
        Loads params.yaml
        Returns: yaml object

        """
        with open(self.configPath) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config

    def dumpData(self, filename, data):
        with open(filename, "a+") as f:
            json.dump(data, f, indent=4)


    def savemodel(self, modelName, model, subdir=None):
        """
        Saves model for future processing

        Args:
            modelName: Name of the model to be saved
            model: Model object
            subdir: Name of the subdirectory

        Returns:None

        """

        if subdir is not None:
            self.new_model_directory = os.path.join(self.model_directory, subdir)
            self.path = os.path.join(self.new_model_directory, modelName)
        else:
            self.path = os.path.join(self.model_directory, modelName)
        try:

            if os.path.isdir(self.path):  # remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(self.path)
            else:
                os.makedirs(self.path)
        except Exception as e:
            print(str(e))

        joblib.dump(model, self.path + '/' + modelName + '.pkl')

    def removeDir(self, path):
        """
        Removes a directory
        Args:
            path: Path of the directory to be removed

        Returns: None

        """
        if os.path.exists(path):
            shutil.rmtree(path)

    def loadModel(self, modelName, modelFolder=None):
        """
        Loads earlier saved model
        Args:
            modelName: Name of the model to be loaded
            modelFolder: Folder of the mode where it is stored

        Returns:
            model: loaded model object

        """
        if modelFolder is not None:
            self.path = os.path.join(self.model_directory, modelFolder)
        else:
            self.path = os.path.join(self.model_directory, modelName)
        try:
            model = joblib.load(self.path + '/' + modelName + '.pkl')
        except Exception as e:
            print(str(e))
        return model

    def archiveData(self, src):

        try:
            for i in src:
                filePath = os.path.join(src, i)
                shutil.move(filePath, 'Data/archivedData')
        # This exception is raised when a system function returns a system - related error.
        except OSError as error:
            print(error)

        # Exception if both source and destination files are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")

        # For other errors
        except Exception as e:
            print("Error occurred while copying file.", e)

