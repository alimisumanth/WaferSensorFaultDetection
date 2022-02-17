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
import pandas as pd
from sklearn.model_selection import train_test_split
from MLAlgo.clustering import clustering
from MLAlgo.classification import classification
from sklearn.metrics import roc_auc_score, accuracy_score
from Utils import Utils
import os


class modelTuner:
    """
    during training input data is trained with different machineLearning model and
    best performing model selected from them. While predicting the saved best performing model is
    loaded and input is passed into it for prediction

    Attributes:
        features: input data

    """

    def __init__(self):
        self.features = None
        self.clusters = None
        self.predicted_data = pd.DataFrame()
        self.clustering = clustering.clustering()
        self.classification = classification.WaferClassification()
        self.utils = Utils.utils()

    def get_best_model(self, features, labels):
        """
        Input data is divided into different clusters, each cluster is trained with different
        MachineLearning models and best performing model is selected.
        Args:
            features: input features
            labels: target data

        Returns: None

        """
        self.features = self.clustering.KMeansAlgo(features)
        self.features['labels'] = labels

        for i in self.features['clusters'].unique():
            cluster = self.features[features['clusters'] == i]
            cluster_features = cluster.drop(['clusters', 'labels'], axis=1)
            cluster_label = cluster['labels']
            x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=0.3)
            xgb = self.classification.XgBoostClassifier(x_train, y_train)
            y_predict = xgb.predict(x_test)
            # if there is only one label in y, then roc_auc_score returns error. We
            # will use accuracy in that case
            if len(y_test.unique()) == 1:
                xgb_score = accuracy_score(y_test, y_predict)
            else:
                xgb_score = roc_auc_score(y_test, y_predict)
            rf = self.classification.RandomForestClassifier(x_train, y_train)
            y_predict = rf.predict(x_test)
            # if there is only one label in y, then roc_auc_score returns error. We
            # will use accuracy in that case
            if len(y_test.unique()) == 1:
                rf_score = accuracy_score(y_test, y_predict)
            else:
                rf_score = roc_auc_score(y_test, y_predict)
            if rf_score > xgb_score:
                self.utils.savemodel("randomForest_" + str(i), rf, 'classification')
            else:
                self.utils.savemodel("XGBoost_" + str(i), xgb, 'classification')

    def findModels(self, cluster):
        """
        finds the model on which the cluster is trained
        Args:
            cluster: cluster id

        Returns: Cluster folder name

        """
        path = 'models/classification'
        for i in os.listdir(path):
            if os.path.isdir(path + '/' + i):
                if int(i.split('_')[1]) == int(cluster):
                    return i

    def predictData(self, features):
        """
        Saved model is loaded and input features are passed into it for prediction
        Args:
            features: input data

        Returns: predicted data

        """
        self.clusters = self.clustering.getClusters(features)

        for i in features['clusters'].unique():
            cluster = features[features['clusters'] == i]
            cluster_features = cluster.drop(['clusters'], axis=1)
            ModelName = self.findModels(i)
            model = self.utils.loadModel(ModelName, 'classification/' + ModelName)
            output = model.predict(cluster_features)

            cluster_features['prediction'] = output
            self.predicted_data = self.predicted_data.append(cluster_features, ignore_index=True)

        return self.predicted_data
