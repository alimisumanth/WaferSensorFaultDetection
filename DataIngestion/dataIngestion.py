from DataPreProcessing import PreProcessing
from Utils import Utils
from InputValidation import InputValidation
from databaseOperations import databaseOperations
import os
import shutil


class DataIngestion:
    def __init__(self):
        self.preprocessing = PreProcessing.PreProcessing()
        self.utils = Utils.utils()
        self.inputValidation = InputValidation.inputValidation()
        self.dbOperations = databaseOperations.Database()

    def rawDataLocal(self, path):
        rawdata = 'Data/rawData'
        self.utils.dircheck(rawdata)
        files = [i for i in os.listdir(path) if i.endswith('.csv')]
        for i in files:
            srcpath = os.path.join(path, i)
            shutil.copy(srcpath, rawdata)

    def LoadToDB(self):
        self.inputValidation.Filevalidation()
        self.inputValidation.columnValidation()
        session = self.dbOperations.DBConnection()
        self.dbOperations.LoadtoDB(session)
        session.close()

    def LoadFromDB(self):
        session = self.dbOperations.DBConnection()
        dataFrame = self.dbOperations.LoadFromDB(session)
        session.close()
        return dataFrame
