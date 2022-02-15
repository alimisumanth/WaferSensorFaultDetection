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
    Validates input data at file level and data level according to data sharing agreement. At file level it checks
    for file name format and at data level it checks for number of columns, data type of columns.

    Attributes:
        configPath: Data sharing agreement files used for training and prediction
        rawData:    Path where raw data files are stored
        goodData:   path where validated files are stored
        badData:    Path where corrupted files are stored

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
        self.configPath = ''
        self.goodData = 'Data/goodData'
        self.badData = 'Data/BadData'
        self.rawdata = 'Data/rawData'

    def Filevalidation(self):
        """
        It collects data from Data/rawData path and filters csv files from the path and checks if the file name is
        according to the data sharing agreement or not. If file name does not match with data sharing agreement then
        it transfers the data to bad data folder and if it matches the data sharing agreement then it will be
        transferred/moved to good data folder.

        Returns: None

        Raises:
            OSError: system related errors including file I/O like file not found etc.
            SameFileError: When source and destination files are same
            Exception: raised for other errors

        """

        files = [i for i in os.listdir(self.rawdata) if i.endswith('.csv')]

        # collecting regex for file name validation
        regex = self.preProcessing.regexMatching()

        # Directory Creation
        self.utils.dircheck(self.goodData)
        self.utils.dircheck(self.badData)

        # Iterating over files in rawData path
        for i in files:
            try:
                srcPath = os.path.join(self.rawdata, i)
                if re.match(regex, i) is not None:  # Regex matching with file name
                    shutil.move(srcPath, self.goodData, copy_function=shutil.copy)  # Moving files to goodData path
                else:
                    shutil.move(srcPath, self.badData, copy_function=shutil.copy)  # Moving files to badData path

            # This exception is raised when a system function returns a system - related error.
            except OSError as error:
                print(error)

            # Exception if both source and destination files are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")

            # For other errors
            except Exception as e:
                print("Error occurred while copying file.", e)

    def columnValidation(self, state):
        """
        It collects data from Data/goodData path and iterates over each file. For each file it checks number of
        columns in the file, data type of the columns. If they are not according to the data sharing agreement then
        they will be moved to badData folder.

        Args:
            state: Mode of validation

        Returns: None

        Raises:
            OSError: system related errors including file I/O like file not found etc.
            SameFileError: When source and destination files are same
            Exception: raised for other errors
        """

        # collecting configPath based on mode of validation train/prediction
        if state == 'train':
            self.configPath = 'schema_training.json'
        else:
            self.configPath = 'schema_prediction.json'

        # Loading config file
        config = self.utils.mdm(self.configPath)

        # Iterating over files in goodData path
        for i in os.listdir(self.goodData):
            try:
                srcPath = os.path.join(self.goodData, i)
                df = pd.read_csv(srcPath)
                if len(df.columns) != config["NumberofColumns"]: # Check for number of columns
                    shutil.move(srcPath, self.badData, copy_function=shutil.copy)

            # This exception is raised when a system function returns a system - related error.
            except OSError as error:
                print(error)

            # Exception if both source and destination files are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")

            # For other errors
            except Exception as e:
                print("Error occurred while copying file.", e)
