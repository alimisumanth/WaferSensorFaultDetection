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


class clustering:
    """
    This class helps to group input data into different clusters
    """

    def __init__(self):
        self.wcss = None
        self.utils = utils()

    def find_optimal_k(self, features):
        """
        Finding Optimal value of k using KneeLocater method
        Args:
            features: Input features

        Returns: optimal K value

        """
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
        return knn

    def KMeansAlgo(self, features):
        """
        Input data is grouped into different clusters.
        Args:
            features: Input features to be grouped

        Returns: clustered dataframe

        """
        k_means = self.find_optimal_k(features)
        k = k_means.knee
        clusters = KMeans(n_clusters=k, init='k-means++', random_state=42)
        res = clusters.fit_predict(features)
        self.utils.savemodel("KMeans", clusters, 'clustering')
        features['clusters'] = res
        return features

    def getClusters(self, features):
        """
        find the cluster number of input data using saved model
        Args:
            features: Input data

        Returns: clustered data

        """
        kMeans = self.utils.loadModel("KMeans", "clustering/KMeans")
        clusters = kMeans.predict(features)
        features['clusters'] = clusters
        return features
