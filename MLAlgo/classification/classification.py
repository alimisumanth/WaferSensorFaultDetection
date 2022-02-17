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
        self.config = None
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

        self.config = self.utils.loadYaml()


        self.param_grid = {
            "n_estimators": self.config['estimators']['RandomForestClassifier']['params']['n_estimators'],
            "criterion": self.config['estimators']['RandomForestClassifier']['params']['criterion'],
            "max_depth": self.config['estimators']['RandomForestClassifier']['params']['max_depth'],
            "max_features": self.config['estimators']['RandomForestClassifier']['params']['max_features']
        }



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
                                             max_features=max_features)
        final_model.fit(x, y)

        params = self.model.best_params_
        return final_model, params

    def XgBoostClassifier(self, x, y):
        """
        Input data is trained with XGBoost Classification model.  Grid Search technique for finding the optimal
        hyperparameters to increase the model performance.
        Args:
            x: features
            y: labels

        Returns:
            XGB: XGBoost classifier object


        """
        self.config = self.utils.loadYaml()

        self.param_grid = {

            'learning_rate': self.config['estimators']['XGBoostClassifier']['params']['learning_rate'],
            'max_depth': self.config['estimators']['XGBoostClassifier']['params']['max_depth'],
            'n_estimators': self.config['estimators']['XGBoostClassifier']['params']['n_estimators']

        }
        self.estimator = XGBClassifier()
        XGB_GCV = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid, cv=5, verbose=3)
        XGB_GCV.fit(x, y)
        self.learning_rate = XGB_GCV.best_params_['learning_rate']
        self.max_depth = XGB_GCV.best_params_['max_depth']
        self.n_estimators = XGB_GCV.best_params_['n_estimators']

        params = XGB_GCV.best_params_
        XGB = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators, eval_metric='mlogloss')
        XGB.fit(x, y)

        return XGB, params

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
