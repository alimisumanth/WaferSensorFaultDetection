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
from WaferLogging import WaferLogging
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
        processedData:   path where validated files are stored
        corruptedData:    Path where corrupted files are stored

    Methods:
    Filevalidation():
        it collects data from Data/rawData path and iterates over each file and compares if file name
        is matching with regular expression.
    columnValidation():
         It collects data from Data/processedData path and iterates over each file and checks if the number of columns
         and date type of columns matches according to the data sharing agreement or not

    """

    def __init__(self):
        self.inputValLogger = None
        self.configPath = ''
        self.processedData = 'Data/processedData'
        self.corruptedData = 'Data/corruptedData'
        self.rawdata = 'Data/rawData'
        self.preProcessing = PreProcessing.PreProcessing()
        self.utils = Utils.utils()
        self.waferLogger = WaferLogging.WaferLogging()


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
        self.inputValLogger = self.waferLogger.getLogger('inputValidation')
        self.inputValLogger.info('File validation started')
        self.inputValLogger.info('Filtering files ends with csv format')
        files = [i for i in os.listdir(self.rawdata) if i.endswith('.csv')]

        self.inputValLogger.info('get file name regex')
        # collecting regex for file name validation
        regex = self.preProcessing.regexMatching()
        self.inputValLogger = self.waferLogger.getLogger('inputValidation')
        # Directory Creation
        self.inputValLogger.info('Create preProcessedData directory if not exists')
        self.utils.dirCheck(self.processedData)
        self.inputValLogger.info('Create corruptedData directory if not exists')
        self.utils.dirCheck(self.corruptedData)

        self.inputValLogger.info('Iterating over files in raw data directory')
        # Iterating over files in rawData path
        for i in files:
            try:
                srcPath = os.path.join(self.rawdata, i)
                if re.match(regex, i) is not None:  # Regex matching with file name
                    shutil.move(srcPath, self.processedData, copy_function=shutil.copy)  # Moving files to
                    # processedData path
                    self.inputValLogger.info('moving '+i+'from '+'rawData directory to processedData directory')
                else:
                    shutil.move(srcPath, self.corruptedData, copy_function=shutil.copy)  # Moving files to
                    # corruptedData path
                    self.inputValLogger.info('File name not matching, moving ' + i + 'from ' + 'rawData directory to '
                                                                                               'corruptedData '
                                                                                               'directory')

            # This exception is raised when a system function returns a system - related error.
            except OSError as error:
                self.inputValLogger.error(error)

            # Exception if both source and destination files are same
            except shutil.SameFileError:
                self.inputValLogger.error("Source and destination represents the same file.")

            # For other errors
            except Exception as e:
                self.inputValLogger.exception("Error occurred while copying file.", e)
        self.inputValLogger.info('File validation completed')

    def columnValidation(self, state):
        """
        It collects data from Data/processedData path and iterates over each file. For each file it checks number of
        columns in the file, data type of the columns. If they are not according to the data sharing agreement then
        they will be moved to corruptedData folder.

        Args:
            state: Mode of validation

        Returns: None

        Raises:
            OSError: system related errors including file I/O like file not found etc.
            SameFileError: When source and destination files are same
            Exception: raised for other errors
        """
        self.inputValLogger = self.waferLogger.getLogger('inputValidation')
        self.inputValLogger.info('columnValidation started')
        # collecting configPath based on mode of validation train/prediction
        if state == 'training':
            self.configPath = 'schema_training.json'
        else:
            self.configPath = 'schema_prediction.json'
        self.inputValLogger.info('Loading master data management file')
        # Loading config file
        config = self.utils.mdm(self.configPath)

        self.inputValLogger.info('Iterating over files from processedData folder')
        # Iterating over files in processedData path
        for i in os.listdir(self.processedData):
            try:
                srcPath = os.path.join(self.processedData, i)
                df = pd.read_csv(srcPath)
                if len(df.columns) != config["NumberofColumns"]: # Check for number of columns
                    self.inputValLogger.info('Number of columns are not matching, moving '+i+' to corruptedData folder')
                    shutil.move(srcPath, self.corruptedData, copy_function=shutil.copy)

            # This exception is raised when a system function returns a system - related error.
            except OSError as error:
                self.inputValLogger.error(error)

            # Exception if both source and destination files are same
            except shutil.SameFileError:
                self.inputValLogger.error("Source and destination represents the same file.")

            # For other errors
            except Exception as e:
                self.inputValLogger.exception("Error occurred while copying file.", e)
