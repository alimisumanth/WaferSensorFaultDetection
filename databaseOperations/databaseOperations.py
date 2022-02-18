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
from WaferLogging import WaferLogging
import sqlite3
import os
import pandas as pd


class Database:
    """
        Description: Database class contains various operations performed on sqlite database

        Attributes:
            session: DB session object
            processedData: Path to validated files folder

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
        self.waferLogger = WaferLogging.WaferLogging()
        self.dbLogger = self.waferLogger.getLogger('dbLogger')
        self.session = None
        self.processedData = 'Data/processedData'

    def DBConnection(self):
        """
        Creates a new sqlite database connection to db.sqlite3 database
        Returns: sqlite object
        Raises: sqlite3 error
        """
        # creating a new database connection
        try:
            self.dbLogger.info('Creating a Database connection')
            self.session = sqlite3.connect('db.sqlite3')
        except sqlite3.Error as e:
            self.dbLogger.error('Exception occurred: ', str(e))

        return self.session

    def insertIntoDB(self, session, state):
        """
        Reads files in Data/processedData directory and generates a dataframe.
        Generated dataframe is stored in database which can be used later for further processing of data

        Args:
         session: sqlite database connection object
         state: Mode of loading data to database

        Raises:
          Integrity Error: raises error when unique constraint is violated
          Exception: For generic error handling

        Returns: None

        """
        self.dbLogger.info("Inserting processedData into database")
        self.dbLogger.info("Inserting processedData into database")
        self.utils.dirCheck(self.processedData)

        files = [i for i in os.listdir(self.processedData)]
        self.dbLogger.info("Iterating over files in processedData directory and "
                           "appending its data to a dataframe")
        dataframe = pd.read_csv(os.path.join(self.processedData, files[0]), index_col=0)
        for file in files[1:]:
            df = pd.read_csv(os.path.join(self.processedData, file), index_col=0)
            dataframe = dataframe.append(df, ignore_index=True)
        try:
            if state == 'training':
                self.dbLogger.info("processed data loaded to wafer_train table")
                dataframe.to_sql('wafer_train', session, index=False, if_exists='replace')
            elif state == 'prediction':
                dataframe.to_sql('wafer_predict', session, index=False, if_exists='replace')
        except sqlite3.IntegrityError as e:
            self.dbLogger.error('Unique constraint violate:', str(e))
        except Exception as e:
            self.dbLogger.exception('Exception occurred:', str(e))

    def retrieveFromDB(self, session, state):
        """
        Loads data from the database created using insertIntoDB method for further processing of data

        Args:
         session: sqlite database connection object
         state: Mode of loading data from database

        Returns: wafer data from database(dataframe)
        """

        try:
            if state == 'training':
                self.dbLogger.info('Retrieving data from wafer_train table')
                df = pd.read_sql_query("select * from wafer_train", session)
            elif state == 'prediction':
                df = pd.read_sql_query("select * from wafer_predict", session)
        except Exception as e:
            self.dbLogger.exception('Exception occurred:', str(e))

        return df

    def dropTable(self, session):
        """
        Drops an existing table from the database

        Args session: sqlite database connection object

        Returns: None
        """
        try:
            self.dbLogger.info('dropping wafer_train table')
            query = 'DROP TABLE IF EXISTS  wafer'
            session.execute(query)
        except Exception as e:
            self.dbLogger.exception('Exception occurred', str(e))
