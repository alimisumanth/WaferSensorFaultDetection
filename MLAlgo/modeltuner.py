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
from WaferLogging import WaferLogging
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
        self.reports = None
        self.params_path = None
        self.scores_path = None
        self.params = {}
        self.scores = {}
        self.predicted_data = pd.DataFrame()
        self.clustering = clustering.clustering()
        self.classification = classification.WaferClassification()
        self.waferLogger = WaferLogging.WaferLogging()
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
        logger = self.waferLogger.getLogger('trainingPhase')
        logger.info('Data is sent to KMeans algorithm for clustering')
        self.features = self.clustering.KMeansAlgo(features)
        self.features['labels'] = labels
        logger = self.waferLogger.getLogger('modelTuner')
        logger.info('Training clusters with RandomForestClassifier and XGBoostClassifier')

        for i in self.features['clusters'].unique():
            logger.info('Tuning cluster: ', str(i))
            cluster = self.features[features['clusters'] == i]
            cluster_features = cluster.drop(['clusters', 'labels'], axis=1)
            cluster_label = cluster['labels']
            logger.info('Splitting train and test data for cluster ', str(i))
            x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=0.3)
            logger.info('Train data is passed into XGBoostClassifier')
            xgb, XGParams = self.classification.XgBoostClassifier(x_train, y_train)
            logger = self.waferLogger.getLogger('modelTuner')
            logger.info('Test data is sent for prediction')
            y_predict = xgb.predict(x_test)
            # if there is only one label in y, then roc_auc_score returns error. We
            # will use accuracy in that case

            if len(y_test.unique()) == 1:
                logger.info('calculating accuracy score')
                xgb_score = accuracy_score(y_test, y_predict)
            else:
                logger.info('calculating roc_auc score')
                xgb_score = roc_auc_score(y_test, y_predict)
            logger.info(f'XGBoost model score: {xgb_score}')

            logger.info('Train data is passed into RandomForestClassifier')
            rf, RFParams = self.classification.RandomForestClassifier(x_train, y_train)
            logger = self.waferLogger.getLogger('modelTuner')
            logger.info('sending test data for prediction')
            y_predict = rf.predict(x_test)
            # if there is only one label in y, then roc_auc_score returns error. We
            # will use accuracy in that case
            if len(y_test.unique()) == 1:
                logger.info('calculating accuracy score')
                rf_score = accuracy_score(y_test, y_predict)
            else:
                logger.info('calculating roc_auc score')
                rf_score = roc_auc_score(y_test, y_predict)
            logger.info(f'RandomForestClassifier model score: {rf_score}')
            logger.info('comparing XGBoostClassifierScore and RandomForestClassifier')
            if rf_score > xgb_score:
                logger.info(f'saving RandomForestClassifier for cluster {str(i)}')
                self.utils.savemodel("randomForest_" + str(i), rf, 'classification')
                self.params['cluster_' + str(i)] = {'model': 'RandomForestClassifier', 'params_path': RFParams}
                self.scores['cluster_' + str(i)] = {'model': 'RandomForestClassifier', 'score': rf_score}

            else:
                logger.info(f'saving XGBoostClassifier for cluster {str(i)}')
                self.utils.savemodel("XGBoost_" + str(i), xgb, 'classification')
                self.params['cluster_' + str(i)] = {'model': 'XGBoostClassifier', 'params_path': XGParams}
                self.scores['cluster_' + str(i)] = {'model': 'XGBoostClassifier', 'score': xgb_score}

        self.reports = self.utils.loadYaml()
        self.params_path = self.reports['reports']['params']
        self.scores_path = self.reports['reports']['scores']
        self.utils.removeDir('reports')
        self.utils.dirCheck('reports')
        logger.info('saving scores and params of all clusters')
        self.utils.dumpData(self.scores_path, self.scores)
        self.utils.dumpData(self.params_path, self.params)

    def findModels(self, cluster):
        """
        finds the model on which the cluster is trained
        Args:
            cluster: cluster id

        Returns: Cluster folder name

        """
        logger = self.waferLogger.getLogger('modelTuner')
        logger.info('finding best performing model of a cluster based on id')
        path = 'models/classification'
        for i in os.listdir(path):
            if os.path.isdir(path + '/' + i):
                if int(i.split('_')[1]) == int(cluster):
                    logger.info(f'cluster {str(i)} saved model is {i}')
                    return i

    def predictData(self, features):
        """
        Saved model is loaded and input features are passed into it for prediction
        Args:
            features: input data

        Returns: predicted data

        """
        logger = self.waferLogger.getLogger('modelTuner')
        logger.info('Predicting data')
        logger.info('Get clusters')
        self.clusters = self.clustering.getClusters(features)
        logger = self.waferLogger.getLogger('modelTuner')
        logger.info('Filtering data based on clusters')
        for i in features['clusters'].unique():
            cluster = features[features['clusters'] == i]
            cluster_features = cluster.drop(['clusters'], axis=1)
            logger.info(f'Finding saved mode for cluster {str(i)}')
            ModelName = self.findModels(i)
            logger.info(f'cluster {str(i)} saved model is {ModelName}')
            logger.info(f'Loading {ModelName}')
            model = self.utils.loadModel(ModelName, 'classification/' + ModelName)
            logger.info('Predicting data')
            output = model.predict(cluster_features)

            cluster_features['prediction'] = output
            logger.info('Returning predicted data')
            self.predicted_data = self.predicted_data.append(cluster_features, ignore_index=True)

        return self.predicted_data
