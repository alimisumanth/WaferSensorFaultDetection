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
from WaferLogging import WaferLogging
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
        self.dataIngestionLogger = None
        self.rawData = 'Data/rawData'
        self.preprocessing = PreProcessing.PreProcessing()
        self.utils = Utils.utils()
        self.inputValidation = InputValidation.inputValidation()
        self.dbOperations = databaseOperations.Database()
        self.waferLogging = WaferLogging.WaferLogging()

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
        self.dataIngestionLogger = self.waferLogging.getLogger('dataIngesion')
        self.dataIngestionLogger.info('Raw data copying to local path')
        try:
            self.dataIngestionLogger.info('Creating a raw data directory if not exists')
            self.utils.dirCheck(self.rawData)
            self.dataIngestionLogger.info('Filtering data files')
            files = [i for i in os.listdir(path) if i.endswith('.csv')]
            self.dataIngestionLogger.info('File copying started')
            for i in files:
                srcPath = os.path.join(path, i)
                self.dataIngestionLogger.info(i + " copied to " + str(self.rawData))
                shutil.copy(srcPath, self.rawData)
            self.dataIngestionLogger.info('File copying completed')

        # Exception if both source and destination files are same
        except shutil.SameFileError:
            self.dataIngestionLogger.error("Source and destination represents the same file.")

        # If permission is denied for copying file
        except PermissionError:
            self.dataIngestionLogger.error("Permission denied.")

        # For other errors
        except Exception as e:
            self.dataIngestionLogger.exception("Error occurred while copying file.", e)

    def LoadToDB(self, state):
        """
        validation at file and column level and transfers data into database

        Returns: None
        """
        self.dataIngestionLogger = self.waferLogging.getLogger('dataIngesion')
        self.dataIngestionLogger.info('Loading data into database')
        logger = self.waferLogging.getLogger('trainingPhase')
        try:
            logger.info('File validation started')
            self.inputValidation.Filevalidation()
            logger = self.waferLogging.getLogger('trainingPhase')
            logger.info('File validation ended')
            logger.info('Column validation started')
            self.inputValidation.columnValidation(state)
            logger = self.waferLogging.getLogger('trainingPhase')
            logger.info('Column validation ended')
            session = self.dbOperations.DBConnection()
            logger = self.waferLogging.getLogger('trainingPhase')
            logger.info('Data Insertion into  database started')
            self.dbOperations.insertIntoDB(session, state)
            logger = self.waferLogging.getLogger('trainingPhase')
            logger.info('Data Insertion into  database ended')
            logger.info('Raw data folder removed')
            self.utils.removeDir('Data/rawData')
            logger.info('Processed data folder removed')
            self.utils.removeDir('Data/processedData')
            session.close()
            self.dataIngestionLogger = self.waferLogging.getLogger('dataIngesion')
            self.dataIngestionLogger.info('Loading data into database completed')
        except Exception as e:
            logger.exception(e)

    def LoadFromDB(self, state):
        """
        Loads data from database

        Returns: dataframe
        """
        self.dataIngestionLogger = self.waferLogging.getLogger('dataIngesion')
        self.dataIngestionLogger.info('Loading data from database started')
        session = self.dbOperations.DBConnection()
        dataFrame = self.dbOperations.retrieveFromDB(session, state)
        self.dataIngestionLogger = self.waferLogging.getLogger('dataIngesion')
        self.dataIngestionLogger.info('Loading data from database ended')
        session.close()
        return dataFrame
