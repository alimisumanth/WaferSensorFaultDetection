from DataPreProcessing import PreProcessing
from Utils import Utils
import os
import re
import pandas as pd
import shutil


class inputValidation:
    def __init__(self):
        self.preProcessing = PreProcessing.PreProcessing()
        self.utils = Utils.utils()
        pass

    def Filevalidation(self):
        path = 'Data/rawData'
        files = [i for i in os.listdir(path) if i.endswith('.csv')]
        regex = self.preProcessing.regexMatching()
        GoodData = 'Data/GoodData'
        BadData = 'Data/BadData'
        self.utils.dircheck(GoodData)
        self.utils.dircheck(BadData)
        for i in files:
            srcpath = os.path.join(path
                                   , i)
            if re.match(regex, i) is not None:
                shutil.move(srcpath, GoodData,copy_function=shutil.copy)
            else:
                shutil.move(srcpath, BadData, copy_function=shutil.copy)
    def columnValidation(self):
        path = 'Data/goodData'
        BadData = 'Data/BadData'
        configpath = 'schema_training.json'
        config = self.utils.mdm(configpath)
        for i in os.listdir(path):
            srcpath = os.path.join(path, i)
            df = pd.read_csv(srcpath)
            if len(df.columns) != config["NumberofColumns"]:
                shutil.move(srcpath, BadData, copy_function=shutil.copy)



