# -*- coding: utf-8 -*-

"""
=============================================================================
Created on: 08-02-2022 11:23 PM
Created by: ASK
=============================================================================

Project Name: WaferSensorFaultDetection

File Name: databaseOperations.py

Description : databaseOperations contains various operations performed on database.

Version: 1.0

Revision: None

=============================================================================
"""

# importing required libraries
from Utils import Utils
import sqlite3
import os
import pandas as pd


class Database:
    """
        Description: Database class contains various operations performed on sqlite database

        Attributes:
            session: DB session object
            goodData: Path to validated files folder

        Methods:

        DBConnection():
            Creates a new connection to the database
        LoadToDB(session):
            Loads data into database
        LoadFromDB(session):
            Loads data from database
        dropTable(session):
            drops table from database

    """

    def __init__(self):
        """
            Constructor initializes Utils object
        """

        self.utils = Utils.utils()
        self.session = None
        self.goodData = 'Data/goodData'

    def DBConnection(self):
        """
        Creates a new sqlite database connection to db.sqlite3 database
        :return: sqlite object
        """
        # creating a new database connection
        self.session = sqlite3.connect('db.sqlite3')
        return self.session

    def LoadToDB(self, session, state):
        """
        Reads files in Data/goodData directory and generates a dataframe.
        Generated dataframe is stored in database which can be used later for further processing of data

        Args:
         session: sqlite database connection object

        :return: None

        """

        self.utils.dircheck(self.goodData)
        files = [i for i in os.listdir(self.goodData)]
        dataframe = pd.read_csv(os.path.join(self.goodData, files[0]), index_col=0)
        for file in files[1:]:
            df = pd.read_csv(os.path.join(self.goodData, file), index_col=0)
            dataframe = dataframe.append(df, ignore_index=True)
        if state == 'train':
            dataframe.to_sql('wafer_train', session, index=False, if_exists='replace')
        elif state == 'predict':
            dataframe.to_sql('wafer_predict', session, index=False, if_exists='replace')

    def LoadFromDB(self, session, state):
        """
        Loads data from the database created using LoadToDB method for further processing of data

        :param session: sqlite database connection object

        :return: wafer data from database(dataframe)
        """
        if state == 'train':
            df = pd.read_sql_query("select * from wafer_train", session)
        elif state == 'predict':
            df = pd.read_sql_query("select * from wafer_predict", session)

        return df

    def dropTable(self, session):
        """
        Drops an existing table from the database

        :param session: sqlite database connection object

        :return: None
        """
        query = 'DROP TABLE IF EXISTS  wafer'
        session.execute(query)
