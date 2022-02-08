from Utils import Utils
import sqlite3
import os
import pandas as pd


class Database:
    def __init__(self):
        self.utils = Utils.utils()

    def DBConnection(self):
        session = sqlite3.connect('db.sqlite3')
        return session

    def LoadtoDB(self, session):
        goodData = 'Data/goodData'
        self.utils.dircheck(goodData)
        files = [i for i in os.listdir(goodData)]
        dataframe = pd.read_csv(os.path.join(goodData, files[0]), index_col=0)
        for file in files[1:]:
            df = pd.read_csv(os.path.join(goodData, file), index_col=0)
            dataframe = dataframe.append(df, ignore_index=True)
        dataframe.to_sql('wafer', session, if_exists='replace')

    def LoadFromDB(self, session):
        df = pd.read_sql_query("select * from wafer", session)
        return df

    def dropTable(self, session):
        query = 'DROP TABLE IF EXISTS  wafer'
        session.execute(query)
