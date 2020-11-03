#!/usr/bin/env python3
"""Initialize K-means 2"""
import numpy as np


def init(X, k):
    """initializes cluster centroids for K-means:

    -> X is a numpy.ndarray of shape (n, d) containing the dataset that
        will be used for K-means clustering
        * n is the number of data points
        * d is the number of dimensions for each data point

    -> k is a positive integer containing the number of clusters

    -> Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    n, d = X.shape
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)

    centroids = np.random.uniform(low=0., high=1., size=(k, d))
    centroids = centroids * (x_max - x_min) + x_min

    return centroids


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset:

    -> X is a numpy.ndarray of shape (n, d) containing the dataset
        * n is the number of data points
        * d is the number of dimensions for each data point

    -> k is a positive integer containing the number of clusters

    -> iterations is a positive integer containing the maximum number of
        iterations that should be performed

    -> Returns: C, clss, or None, None on failure
        * C is a numpy.ndarray of shape (k, d) containing the centroid
            means for each cluster
        * clss is a numpy.ndarray of shape (n,) containing the index of
            the cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    centroids = init(X, k)
    clss = 0

    for i in range(iterations):
        substrac = X[:, None] - centroids
        D = np.linalg.norm(substrac, axis=-1)
        clss = np.argmin(D, axis=-1)

        cpy = np.copy(centroids)

        for c in range(k):
            idx = np.where(clss == c)[0]
            if len(idx) == 0:
                centroids[c] = init(X, 1)
            else:
                centroids[c] = np.mean(X[idx], axis=0)

        if (cpy == centroids).all():
            return centroids, clss

    substrac = X[:, None] - centroids
    D = np.linalg.norm(substrac, axis=-1)
    clss = np.argmin(D, axis=-1)

    return centroids, clss
