# -*- coding: utf-8 -*-

"""
=============================================================================
Created on: 08-02-2022 11:23 PM
Created by: ASK
=============================================================================

Project Name: WaferSensorFaultDetection

File Name: PreProcessing.py

Description : This module contains various kinds of preprocessing techniques

Version: 1.0

Revision: None

=============================================================================
"""

# importing required libraries
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from Utils import Utils
from WaferLogging import WaferLogging

class PreProcessing:
    """
        Data Preprocessing techniques

        Attributes:utils

        Methods:

        regexMatching()
           Return a regex used for checking file name
        nullValueCheck(df)
            checks for null values in dataframe
        KNNImputer(df)
            Imputes missing values using KNN imputer
        zerovarcol(df)
            drops columns with zero variance

    """

    def __init__(self):
        self.regex = ''
        self.utils = Utils.utils()
        self.waferLogger = WaferLogging.WaferLogging()

    def regexMatching(self, state):
        """
        defines a regex pattern which can be used to match if the file names are as per data sharing agreement or not

        :return: regular expression(regex)
        """
        logger = self.waferLogger.getLogger(str(state)+'_preProcessing')
        self.regex = "['wafer'|'Wafer']+[\_]+(\d{8}\_)+(\d{6})+\.csv"
        logger.info('file name regex: '+self.regex)
        return self.regex

    def nullValueCheck(self, df, state):
        """
        Checks for null values in dataframe

        Args: df - A dataframe

        Returns: Boolean value: if number of null columns is greater than 0 return True else return False
        """
        logger = self.waferLogger.getLogger(str(state)+'_preProcessing')
        nullColumns = [(i, df[i].isnull().sum()) for i in df.columns if df[i].isnull().sum() > 0]
        logger.info('Number of columns with Null values: '+str(len(nullColumns)))
        return len(nullColumns) > 0

    def KNNImputer(self, df, state):
        """
        Imputes missing values using scikit-learn preprocessing module - KNNImputer method

        Args
          state: Mode of Imputation
          df: A dataframe

        Returns: Return a new dataframe whose null values are imputed by KNN Imputer
        """
        logger = self.waferLogger.getLogger(str(state)+'_preProcessing')
        if state == 'training':
            cols = [i for i in df.columns if df[i].dtypes != 'object']
            logger.info('columns with null values: '+' ,'.join(cols))
            logger.info('Using KNNImputer for handling missing values')
            knn = KNNImputer()
            new_df = pd.DataFrame(knn.fit_transform(df[cols]), columns=cols)
            logger.info('Null value Imputation completed')
            logger.info('Saving KNN Imputer to model directory')
            self.utils.savemodel("KNNImputer", knn, 'Imputer')
        elif state == 'prediction':
            logger.info('Using saved KNN Imputer model for imputing missing values')
            knn = self.utils.loadModel("KNNImputer", 'Imputer/KNNImputer')
            new_df = pd.DataFrame(knn.transform(df), columns=df.columns)
            logger.info('Null Value imputation completed')
        else:
            print('state is not defined')

        return new_df

    def zerovarcol(self, features, state):
        """
        Drops the features whose values are constant or columns having zero variance.

        Args:
          features: list of features whose variance will be checked using VarianceThreshold method
          state: Mode of selection

        Returns: return features which are having non-zero variance
        """
        logger = self.waferLogger.getLogger(str(state)+'_preProcessing')
        logger.info('Removing features with zero variance')
        if state == 'training':
            vt = VarianceThreshold(threshold=0)
            logger.info('fitting data with VarianceThreshold')
            vt.fit(features)
            self.utils.savemodel("VIF", vt)
            logger.info('saving VarianceThreshold model in models directory')
            zeroVarCols = [i for i in features.columns if i not in features.columns[vt.get_support()]]
            logger.info('columns with zero variance: '+','.join(zeroVarCols))
            features.drop(zeroVarCols, axis=1, inplace=True)
            logger.info('columns with zero variance removed')
        elif state == 'prediction':
            try:
                logger.info('Removing features with zero variance using saved VarianceThreshold model')
                vif = self.utils.loadModel("VIF")
                logger.info('VarianceThreshold loaded')
                nonZeroVarCols = [i for i in features.columns if i in features.columns[vif.get_support()]]
                logger.info('columns with variance non zero: ', ','.join(nonZeroVarCols))
                features = pd.DataFrame(vif.transform(features), columns=nonZeroVarCols)
                logger.info('Removed features with zero variance ')
            except Exception as e:
                logger.exception("Exception occurred", str(e))
        return features
