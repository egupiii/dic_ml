from sklearn.datasets import make_blobs
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns



# Create a class of K-means from scratch
class ScratchKMeans():
    """
    Implement K-means from scratch.

    Parameters
    ----------
    k: int
        The number of labels

    num_iter: int
        The number of iteration

    Attributes
    ----------
    centroids: ndarray whose shape is (n_features,n_iters)
        K centroids already fitted
    """

    def __init__(self, k, num_iter):
        # Record hyperparameters as attribute
        self.k = k
        self.iter = num_iter


    def fit(self, X):
        """
        Fit datasets by K-means.

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            Features of train dataset
        """

        # Transform arrays to move their features to rows
        X = X.T

        # Set initial central points
        np.random.seed(32)
        index = np.array(range(X.shape[1]))
        np.random.shuffle(index)
        k_index = index[:self.k]
        self.centroids = X[:, k_index]

        # Update the centroids untill they are changing
        for i in range(self.iter):
            cluster_table = self.assign_cluster(X, self.centroids)
            temporary_centroids = self.update_centroids(X, cluster_table)
            if np.allclose(self.centroids, temporary_centroids):
                break
            else:
                self.centroids = temporary_centroids
        #                 print("iter:",i)

        # Compute the SSE
        return np.sum(np.square(cluster_table))


    # Compute distances between each point and central points and assign them to theirown label
    def assign_cluster(self, X, centroids):
        """
        Assign each data point to the most nearest cluster.

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            Features of train dataset

        centroids: ndarray whose shape is (n_features,n_iters)
            Initial k centroids

        Returns
        ----------
        cluster_table: ndarray whose shape is (n_samples,n_clusters)
            A table showing what cluster each data point belong to and distances between each data point and the most nearest cluster
        """

        cluster_table = np.zeros((X.shape[1], self.k))
        for i in range(X.shape[1]):
            min_d = 1e+10000
            label = 0
            for j in range(self.k):
                d = np.linalg.norm(X[:, i] - centroids[:, j])
                if d < min_d:
                    min_d = d
                    label = j
            cluster_table[i, label] = min_d

        return cluster_table


    # Update centroids
    def update_centroids(self, X, cluster_table):
        """
        Assign each data point to the most nearest cluster.

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            Features of train dataset

        cluster_table: ndarray whose shape is (n_samples,n_clusters)
            A table showing what cluster each data point belong to and distances between each data point and the most nearest cluster

        Returns
        ----------
        centroids: ndarray whose shape is (n_features,n_iters)
            Updated k centroids
        """

        centroids = np.zeros((2, self.k))
        for i in range(self.k):
            index = np.where(cluster_table[:, i] != 0)
            centroids[0, i] = sum(X[0, index[0]]) / len(index[0])
            centroids[1, i] = sum(X[1, index[0]]) / len(index[0])

        return centroids


    def predict(self, X):
        """
        Predict datasets by K-means.

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            Features of train dataset

        Returns
        ----------
        y_pred: ndarray whose shape is (n_samples,n_clusters)
            A table showing what cluster each data point belong to and distances between each data point and the most nearest cluster
        """

        # Transform arrays to move their features to rows
        X = X.T

        # Return a cluster table
        return self.assign_cluster(X, self.centroids)


    # Implement the Elbow method
    def elbow_method(self, X, kinds_of_clusters):
        """
        Implement the Elbow method.

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            Features of train dataset

        kinds_of_clusters: int
            The number of kinds of clusters
        """

        k_list = []
        sse_list = []
        for i in range(kinds_of_clusters + 1):
            kmeans = ScratchKMeans(i + 1, 100)
            sse = kmeans.fit(X)
            k_list.append(i + 1)
            sse_list.append(sse)

        plt.figure(facecolor="azure", edgecolor="coral")
        plt.grid(True)

        plt.plot(k_list, sse_list)

        plt.title("Elbow Method")
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")

        plt.show()