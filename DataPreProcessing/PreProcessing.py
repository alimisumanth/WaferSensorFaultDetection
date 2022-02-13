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

        self.utils = Utils.utils()

    def regexMatching(self):
        """
        defines a regex pattern which can be used to match if the file names are as per data sharing agreement or not

        :return: regular expression(regex)
        """
        regex = "['wafer'|'Wafer']+[\_]+(\d{8}\_)+(\d{6})+\.csv"
        return regex

    def nullValueCheck(self, df):
        """
        Checks for null values in dataframe

        :param df: A dataframe

        :return: Boolean value: if number of null columns is greater than 0 return True else return False
        """
        nullColumns = [(i, df[i].isnull().sum()) for i in df.columns if df[i].isnull().sum() > 0]
        return len(nullColumns) > 0

    def KNNImputer(self, df, state):
        """
        Imputes missing values using scikit-learn preprocessing module - KNNImputer method

        :param state:
        :param df: A dataframe

        :return: Return a new dataframe whose null values are imputed by KNN Imputer
        """
        if state == 'train':
            cols = [i for i in df.columns if df[i].dtypes != 'object']
            knn = KNNImputer()
            new_df = pd.DataFrame(knn.fit_transform(df[cols]), columns=cols)
            self.utils.savemodel("KNNImputer", knn, 'Imputer')
        elif state == 'predict':
            knn = self.utils.loadmodel("KNNImputer")
            new_df = knn.transform(df)
        else:
            pass

        return new_df

    def zerovarcol(self, features, state):
        """
        Drops the features whose values are constant or columns having zero variance.

        :param features: list of features whose variance will be checked using VarianceThreshold method

        :return: return features which are having non-zero variance
        """
        if state == 'train':
            vt = VarianceThreshold(threshold=0)
            vt.fit(features)
            #self.utils.savemodel("VIF", vt)
            zeroVarCols = [i for i in features.columns if i not in features.columns[vt.get_support()]]
            features.drop(zeroVarCols, axis=1, inplace=True)
        elif state == 'predict':
            vif = self.utils.loadmodel("VIF")
            features = vif.transform(features)

        return features
