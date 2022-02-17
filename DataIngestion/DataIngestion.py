# -*- coding: utf-8 -*-

"""
=============================================================================
Created on: 08-02-2022 11:23 PM
Created by: ASK
=============================================================================

Project Name: WaferSensorFaultDetection

File Name: DataIngestion.py

Description : This module contains various kinds of ingestion operations

Version: 1.0

Revision: None

=============================================================================
"""

# importing required libraries

from DataPreProcessing import PreProcessing
from Utils import Utils
from InputValidation import InputValidation
from databaseOperations import databaseOperations
import os
import shutil


class DataIngestion:
    """
    Transfers the data from one source to destination.

    Attributes:

        rawData: contains raw data passed by user

    Methods :

        rawDataLocal(path):
            Transfers data from local source to Data/rawData
        insertIntoDB():
            Transfers data to Database
        retrieveFromDB():
            Transfers data from database

    """

    def __init__(self):
        self.rawData = 'Data/rawData'
        self.preprocessing = PreProcessing.PreProcessing()
        self.utils = Utils.utils()
        self.inputValidation = InputValidation.inputValidation()
        self.dbOperations = databaseOperations.Database()

    def rawDataLocal(self, path):
        """
        Transfers data for the path provided in argument to Data/rawData folder

        Args:
            path: Local folder path where files are available

        Returns:
            None

        Raises:
            SameFileError: If both source and destination files are same
            PermissionError: If permission is denied for copying file
            Exception: If any other exception
        """
        try:

            self.utils.dirCheck(self.rawData)
            files = [i for i in os.listdir(path) if i.endswith('.csv')]
            for i in files:
                srcPath = os.path.join(path, i)
                shutil.copy(srcPath, self.rawData)

        # Exception if both source and destination files are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")

        # If permission is denied for copying file
        except PermissionError:
            print("Permission denied.")

        # For other errors
        except Exception as e:
            print("Error occurred while copying file.", e)

    def LoadToDB(self, state):
        """
        validation at file and column level and transfers data into database

        Returns: None
        """
        try:

            self.inputValidation.Filevalidation()
            self.inputValidation.columnValidation(state)
            session = self.dbOperations.DBConnection()
            self.dbOperations.insertIntoDB(session, state)
            self.utils.removeDir('Data/rawData')
            self.utils.removeDir('Data/GoodData')
            session.close()

        except Exception as e:
            print('Loading to database failed', str(e))

    def LoadFromDB(self, state):
        """
        Loads data from database

        Returns: dataframe
        """
        session = self.dbOperations.DBConnection()
        dataFrame = self.dbOperations.retrieveFromDB(session, state)
        session.close()
        return dataFrame
