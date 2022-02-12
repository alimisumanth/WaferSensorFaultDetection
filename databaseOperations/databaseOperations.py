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

        Attributes: None

        Methods:

        DBConnection():
            Creates a new connection to the database
        LoadtoDB(session):
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

    def DBConnection(self):
        """
        Creates a new sqlite database connection to db.sqlite3 database
        :return: sqlite object
        """
        session = sqlite3.connect('db.sqlite3')
        return session

    def LoadtoDB(self, session):
        """
        Reads files in Data/goodData directory and generates a dataframe.
        Generated dataframe is stored in database which can be used later for further processing of data

        :param session: sqlite database connection object

        :return: None
        """
        goodData = 'Data/goodData'
        self.utils.dircheck(goodData)
        files = [i for i in os.listdir(goodData)]
        dataframe = pd.read_csv(os.path.join(goodData, files[0]), index_col=0)
        for file in files[1:]:
            df = pd.read_csv(os.path.join(goodData, file), index_col=0)
            dataframe = dataframe.append(df, ignore_index=True)
        dataframe.to_sql('wafer', session,index=False, if_exists='replace')

    def LoadFromDB(self, session):
        """
        Loads data from the database created using LoadtoDB method for further processing of data

        :param session: sqlite database connection object

        :return: wafer data from database(dataframe)
        """
        df = pd.read_sql_query("select * from wafer", session)
        return df

    def dropTable(self, session):
        """
        Drops an existing table from the database

        :param session: sqlite database connection object

        :return: None
        """
        query = 'DROP TABLE IF EXISTS  wafer'
        session.execute(query)
