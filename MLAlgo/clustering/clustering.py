# -*- coding: utf-8 -*-
"""
=============================================================================
Created on: 11-02-2022 10:27 PM
Created by: ASK
=============================================================================

Project Name: WaferSensorFaultDetection

File Name: clustering.py

Description:

Version:

Revision:

=============================================================================
"""
from kneed import KneeLocator
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from Utils.Utils import utils
from WaferLogging.WaferLogging import WaferLogging


class clustering:
    """
    This class helps to group input data into different clusters
    """

    def __init__(self):
        self.wcss = None
        self.utils = utils()
        self.waferLogger = WaferLogging()

    def find_optimal_k(self, features):
        """
        Finding Optimal value of k using KneeLocater method
        Args:
            features: Input features

        Returns: optimal K value

        """
        logger = self.waferLogger.getLogger('Training_clustering')
        logger.info('Finding optimal value of K using KneeLocator method')
        self.wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # initializing the KMeans object
            kmeans.fit(features)  # fitting the data to the KMeans Algorithm
            self.wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), self.wcss)  # creating the graph between WCSS and the number of clusters
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('reports/K-Means_Elbow.PNG')  # saving the elbow plot locally
        # finding the value of the optimum cluster programmatically
        knn = KneeLocator(range(1, 11), self.wcss, curve='convex', direction='decreasing')
        logger.info('Optimal value of k : '+str(knn.knee))
        return knn

    def KMeansAlgo(self, features):
        """
        Input data is grouped into different clusters.
        Args:
            features: Input features to be grouped

        Returns: clustered dataframe

        """
        logger = self.waferLogger.getLogger('Training_clustering')
        logger.info('Data Clustering started')
        k_means = self.find_optimal_k(features)
        k = k_means.knee
        logger.info(f'Creating {k} clusters using KMeans algorithm')
        clusters = KMeans(n_clusters=k, init='k-means++', random_state=42)
        res = clusters.fit_predict(features)
        logger.info(f'{k} different clusters created')
        logger.info(f'Saving Kmeans model for production use')
        self.utils.savemodel("KMeans", clusters, 'clustering')
        logger.info(f'Model saved')
        features['clusters'] = res
        logger.info(f'Returning cluster ids')
        return features

    def getClusters(self, features):
        """
        find the cluster number of input data using saved model
        Args:
            features: Input data

        Returns: clustered data

        """
        logger = self.waferLogger.getLogger('Prediction_clustering')
        logger.info('Loading saved Kmeans clustering model')
        kMeans = self.utils.loadModel("KMeans", "clustering/KMeans")
        logger.info('Data is sent to loaded model for creating clusters')
        clusters = kMeans.predict(features)
        features['clusters'] = clusters
        logger.info('Returning cluster ids')
        return features
