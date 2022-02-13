# -*- coding: utf-8 -*-

"""
=============================================================================
Created on: 08-02-2022 11:23 PM
Created by: ASK
=============================================================================

Project Name: WaferSensorFaultDetection

File Name: dataIngestion.py

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

    Methods :

    rawDataLocal(path):
        Transfers data from local source to Data/rawData
    LoadToDB():
        Transfers data to Database
    LoadFromDB():
        Transfers data from database

    """
    def __init__(self):

        self.preprocessing = PreProcessing.PreProcessing()
        self.utils = Utils.utils()
        self.inputValidation = InputValidation.inputValidation()
        self.dbOperations = databaseOperations.Database()

    def rawDataLocal(self, path):
        """
        Transfers data for the path provided in argument to Data/rawData folder

        :param path: Local folder path where files are available

        :return: None
        """
        rawdata = 'Data/rawData'
        self.utils.dircheck(rawdata)
        files = [i for i in os.listdir(path) if i.endswith('.csv')]
        for i in files:
            srcpath = os.path.join(path, i)
            shutil.copy(srcpath, rawdata)

    def LoadToDB(self,state='train'):
        """
        Transfers data into database

        :return: None
        """
        self.inputValidation.Filevalidation()
        self.inputValidation.columnValidation(state)
        session = self.dbOperations.DBConnection()
        self.dbOperations.LoadtoDB(session)
        session.close()

    def LoadFromDB(self):
        """
        Loads data from database

        :return: dataframe
        """
        session = self.dbOperations.DBConnection()
        dataFrame = self.dbOperations.LoadFromDB(session)
        session.close()
        return dataFrame
