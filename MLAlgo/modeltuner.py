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
from sklearn.metrics import roc_auc_score
from  Utils import  Utils


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
            xgb_score = roc_auc_score(y_test, y_predict)
            rf=self.classification.RandomForestClassifier(x_train, y_train)
            y_predict = rf.predict(x_test)
            rf_score =  roc_auc_score(y_test, y_predict)
            if rf_score > xgb_score:
                self.utils.savemodel("randomForest"+str(i), rf)
            else:
                self.utils.savemodel("XGBoost"+str(i), xgb)

