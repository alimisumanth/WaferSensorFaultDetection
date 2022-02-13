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
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from Utils.Utils import utils


class WaferClassification:
    def __init__(self):
        self.utils = utils()

    def RandomForestClassifier(self,x,y):
        """param_grid = {"n_estimators": [50, 100, 130], "criterion": ['gini', 'entropy'],
                      "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}"""
        param_grid = {"n_estimators": [50], "criterion": ['gini', 'entropy'],
                              "max_depth": range(2, 3, 1), "max_features": ['auto']}
        estimator = RandomForestClassifier()
        model=GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=5, verbose=3)
        model.fit(x,y)
        n_estimators = model.best_params_['n_estimators']
        criterion=model.best_params_['criterion']
        max_depth=model.best_params_['max_depth']
        max_features = model.best_params_['max_features']
        final_model=RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,max_features=max_features).fit(x,y)
        return final_model

    def XgBoostClassifier(self,x,y):
        param_grid = {

            'learning_rate': [0.01, 0.001],
            'max_depth': [5, 10],
            'n_estimators': [10, 50]

        }
        estimator=XGBClassifier()
        XGB_GCV=GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5,verbose=3)
        XGB_GCV.fit(x,y)
        learning_rate=XGB_GCV.best_params_['learning_rate']
        max_depth = XGB_GCV.best_params_['max_depth']
        n_estimators = XGB_GCV.best_params_['n_estimators']
        XGB=XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
        XGB.fit(x,y)
        return XGB

    def modelPredictor(self, modelName, features):
        model=self.utils.loadmodel(modelName)
        predictedData=model.predict(features)
        return predictedData









