# -*- coding: utf-8 -*-

"""
=============================================================================
Created on: 08-02-2022 11:23 PM
Created by: ASK
=============================================================================

Project Name: WaferSensorFaultDetection

File Name: InputValidation.py

Description : This module performs validation at file level and data level

Version: 1.0

Revision: None

=============================================================================
"""

# importing required libraries

from DataPreProcessing import PreProcessing
from Utils import Utils
import os
import re
import pandas as pd
import shutil


class inputValidation:
    """
    Validates input data at file level and data level according to data sharing agreement.
    At file level it checks for file name format and at data level it checks for number of columns, data type of columns.

    Methods:

    Filevalidation():
        it collects data from Data/rawData path and iterates over each file and compares if file name
        is matching with regular expression.
    columnValidation():
         It collects data from Data/goodData path and iterates over each file and checks if the number of columns
         and date type of columns matches according to the data sharing agreement or not

    """
    def __init__(self):
        self.preProcessing = PreProcessing.PreProcessing()
        self.utils = Utils.utils()
        pass

    def Filevalidation(self):
        """
        It collects data from Data/rawData path and filters csv files from the path and checks if the file name is
        according to the data sharing agreement or not. If file name doesnot match with data sharing agreement then
        it transfers the data to bad data folder and if it matches thhe data sharing aggrement then it will be
        transferred/moved to good data folder.

        :return: None
        """
        path = 'Data/rawData'
        files = [i for i in os.listdir(path) if i.endswith('.csv')]
        regex = self.preProcessing.regexMatching()
        GoodData = 'Data/GoodData'
        BadData = 'Data/BadData'
        self.utils.dircheck(GoodData)
        self.utils.dircheck(BadData)
        for i in files:
            srcpath = os.path.join(path
                                   , i)
            if re.match(regex, i) is not None:
                shutil.move(srcpath, GoodData, copy_function=shutil.copy)
            else:
                shutil.move(srcpath, BadData, copy_function=shutil.copy)
    def columnValidation(self):
        """
        It collects data from Data/goodData path and iterates over each file. For each file it checks number of
        columns in the file, data type of the columns. If they are not according to the data sharing agreement then
        they will be moved to badData folder.

        :return: None
        """
        path = 'Data/goodData'
        BadData = 'Data/BadData'
        configpath = 'schema_training.json'
        config = self.utils.mdm(configpath)
        for i in os.listdir(path):
            srcpath = os.path.join(path, i)
            df = pd.read_csv(srcpath)
            if len(df.columns) != config["NumberofColumns"]:
                shutil.move(srcpath, BadData, copy_function=shutil.copy)



