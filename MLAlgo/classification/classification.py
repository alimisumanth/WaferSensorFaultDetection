# -*- coding: utf-8 -*-
"""
=============================================================================
Created on: 11-02-2022 11:46 PM
Created by: ASK
=============================================================================

Project Name: WaferSensorFaultDetection

File Name: classification.py

Description:

Version:

Revision:

=============================================================================
"""
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from Utils.Utils import utils


class WaferClassification:
    def __init__(self):
        self.n_estimators = None
        self.max_depth = None
        self.learning_rate = None
        self.model = None
        self.param_grid = None
        self.estimator = None
        self.utils = utils()

    def RandomForestClassifier(self, x, y):
        """
        Input data is trained with Random Forest Classification model.  Grid Search technique for finding the optimal
        hyperparameters to increase the model performance.

        Args:
            x: features
            y: labels

        Returns:
            final_model: Return object of RandomForestClassifier

        """
        # param_grid = {"n_estimators": [50, 100, 130], "criterion": ['gini', 'entropy'],
        #              "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}"""
        self.param_grid = {"n_estimators": [50], "criterion": ['gini', 'entropy'],
                           "max_depth": range(2, 3, 1), "max_features": ['auto']}
        self.estimator = RandomForestClassifier()
        self.model = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid, n_jobs=-1, cv=5, verbose=3)
        self.model.fit(x, y)
        n_estimators = self.model.best_params_['n_estimators']
        criterion = self.model.best_params_['criterion']
        max_depth = self.model.best_params_['max_depth']
        max_features = self.model.best_params_['max_features']
        final_model = RandomForestClassifier(n_estimators=n_estimators,
                                             criterion=criterion,
                                             max_depth=max_depth,
                                             max_features=max_features).fit(x, y)
        return final_model

    def XgBoostClassifier(self, x, y):
        """
        Input data is trained with XGBoost Classification model.  Grid Search technique for finding the optimal
        hyperparameters to increase the model performance.
        Args:
            x: features
            y: labels

        Returns: returns XGBoost classifier object

        """
        self.param_grid = {

            'learning_rate': [0.01, 0.001],
            'max_depth': [5, 10],
            'n_estimators': [10, 50]

        }
        self.estimator = XGBClassifier()
        XGB_GCV = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid, cv=5, verbose=3)
        XGB_GCV.fit(x, y)
        self.learning_rate = XGB_GCV.best_params_['learning_rate']
        self.max_depth = XGB_GCV.best_params_['max_depth']
        self.n_estimators = XGB_GCV.best_params_['n_estimators']
        XGB = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
        XGB.fit(x, y)
        return XGB

    def modelPredictor(self, modelName, features):
        """
        Loads saved model and input features are passed nto it for prediction
        Args:
            modelName: Name of the saved model
            features: input features to be predicted

        Returns: Predicted data

        """
        self.model = self.utils.loadModel(modelName)
        predictedData = self.model.predict(features)
        return predictedData
