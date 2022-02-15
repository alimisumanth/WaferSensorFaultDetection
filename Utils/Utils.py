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

class utils:
    """
    common methods which are used in this project

    Method:

    dircheck(path):
        Creates directories recursively
    mdm(config_path):
        reads and returns master data management file
    """
    def __init__(self):
        self.model_directory = 'models/'


    def dircheck(self, path):
        """
        Creates directories recursively

        :param path: path whose directory is to created

        :return: None
        """
        os.makedirs(path, exist_ok=True)

    def mdm(self, config_path):
        """
        Reads and return master data management file

        :param config_path: path of the master data management file

        :return: master data management file in dic format
        """
        with open(config_path) as file:
                config = json.load(file)
        return config

    def savemodel(self,modelName, model, subdir=None):
        if subdir is not None:
            self.new_model_directory = os.path.join(self.model_directory, subdir)
            self.path=os.path.join(self.new_model_directory, modelName)
        else:
            self.path = os.path.join(self.model_directory, modelName)
        if os.path.isdir(self.path):  # remove previously existing models for each clusters
            shutil.rmtree(self.model_directory)
            os.makedirs(self.path)
        else:
            os.makedirs(self.path)

        joblib.dump(model, self.path+'/'+modelName+'.pkl')

    def removedir(self, path):
        shutil.rmtree(path)

    def loadmodel(self,modelName,modelfolder=None):
        if modelfolder is not None:
            self.path = os.path.join(self.model_directory, modelfolder)
        else:
            self.path = os.path.join(self.model_directory, modelName)
        model=joblib.load(self.path + '/' + modelName + '.pkl')
        return model



