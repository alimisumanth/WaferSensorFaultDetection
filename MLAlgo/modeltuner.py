# -*- coding: utf-8 -*-
"""
=============================================================================
Created on: 11-02-2022 10:10 PM
Created by: ASK
=============================================================================

Project Name: WaferSensorFaultDetection

File Name: modeltuner.py

Description: Finding best performing model using hyperparameter different machine learning models tuning

Version: 1.0

Revision: None

=============================================================================
"""
from sklearn.model_selection import train_test_split
from MLAlgo.clustering import clustering
from MLAlgo.classification import classification
from sklearn.metrics import roc_auc_score,accuracy_score
from Utils import Utils
import os


class modelTuner():
    """

    """
    def __init__(self):
        self.clustering=clustering.clustering()
        self.classification=classification.WaferClassification()
        self.utils = Utils.utils()


    def get_best_model(self,features,labels):
        features = self.clustering.KMeansAlgo(features)
        features['labels'] = labels

        for i in features['clusters'].unique():
            cluster = features[features['clusters'] == i]
            cluster_features = cluster.drop(['clusters', 'labels'], axis=1)
            cluster_label = cluster['labels']
            x_train,x_test,y_train,y_test=train_test_split(cluster_features, cluster_label, test_size=0.3)
            xgb=self.classification.XgBoostClassifier(x_train,y_train)
            y_predict = xgb.predict(x_test)
            if len(y_test.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                xgb_score = accuracy_score(y_test, y_predict)
            else:
                xgb_score = roc_auc_score(y_test, y_predict)
            rf=self.classification.RandomForestClassifier(x_train, y_train)
            y_predict = rf.predict(x_test)
            if len(y_test.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                rf_score = accuracy_score(y_test, y_predict)
            else:
                rf_score = roc_auc_score(y_test, y_predict)
            if rf_score > xgb_score:
                self.utils.savemodel("randomForest_"+str(i), rf, 'classification')
            else:
                self.utils.savemodel("XGBoost_"+str(i), xgb, 'classification')
    def findModels(self,cluster):
        path='models/cluster'
        models=[i for i in os.listdir(path) if os.path.isdir(i)]
        print(models)

        self.utils.loadmodel()

    def predictData(self, features):
        self.clusters=self.clustering.getClusters(features)
        for i in features['clusters'].unique():
            cluster = features[features['clusters'] == i]
            cluster_features = cluster.drop(['clusters'], axis=1)
            self.findModels(i)



