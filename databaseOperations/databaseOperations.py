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
        insertIntoDB(session):
            Loads data into database
        retrieveFromDB(session):
            Loads data from database
        dropTable(session):
            drops table from database

    """

    def __init__(self):
        """
            Constructor initializes
                Utils object
        """

        self.utils = Utils.utils()
        self.session = None
        self.goodData = 'Data/goodData'

    def DBConnection(self):
        """
        Creates a new sqlite database connection to db.sqlite3 database
        Returns: sqlite object
        Raises: sqlite3 error
        """
        # creating a new database connection
        try:
            self.session = sqlite3.connect('db.sqlite3')
        except sqlite3.Error as e:
            print('Exception occurred: ', str(e))

        return self.session

    def insertIntoDB(self, session, state):
        """
        Reads files in Data/goodData directory and generates a dataframe.
        Generated dataframe is stored in database which can be used later for further processing of data

        Args:
         session: sqlite database connection object
         state: Mode of loading data to database

        Raises:
          Integrity Error: raises error when unique constraint is violated
          Exception: For generic error handling

        Returns: None

        """

        self.utils.dirCheck(self.goodData)
        files = [i for i in os.listdir(self.goodData)]
        dataframe = pd.read_csv(os.path.join(self.goodData, files[0]), index_col=0)
        for file in files[1:]:
            df = pd.read_csv(os.path.join(self.goodData, file), index_col=0)
            dataframe = dataframe.append(df, ignore_index=True)
        try:
            if state == 'train':
                dataframe.to_sql('wafer_train', session, index=False, if_exists='replace')
            elif state == 'predict':
                dataframe.to_sql('wafer_predict', session, index=False, if_exists='replace')
        except sqlite3.IntegrityError as e:
            print('Unique constraint violate:', str(e))
        except Exception as e:
            print('Exception occurred:', str(e))

    def retrieveFromDB(self, session, state):
        """
        Loads data from the database created using insertIntoDB method for further processing of data

        Args:
         session: sqlite database connection object
         state: Mode of loading data from database

        Returns: wafer data from database(dataframe)
        """
        try:
            if state == 'train':
                df = pd.read_sql_query("select * from wafer_train", session)
            elif state == 'predict':
                df = pd.read_sql_query("select * from wafer_predict", session)
        except Exception as e:
            print('Exception occurred:', str(e))

        return df

    def dropTable(self, session):
        """
        Drops an existing table from the database

        Args session: sqlite database connection object

        Returns: None
        """
        try:
            query = 'DROP TABLE IF EXISTS  wafer'
            session.execute(query)
        except Exception as e:
            print('Exception occurred', str(e))
