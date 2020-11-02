#!/usr/bin/env python3
"""Initialize K-means 2"""
import numpy as np


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
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)

    labels = np.random.uniform(low=0., high=1., size=(k, n))

    centroids = np.random.uniform(low=0., high=1., size=(k, d))
    centroids = centroids * (x_max - x_min) + x_min

    for i in range(iterations):
        substrac = np.expand_dims(X, 2) - np.expand_dims(centroids.T, 0)
        D = np.linalg.norm(substrac, axis=1)
        clss = np.argmin(D, axis=1)

        if (labels == clss).all():
            labels = clss
            break

        else:
            diff = np.mean(labels != clss)
            labels = clss
            for c in range(k):
                centroids[c] = np.mean(X[labels == c], axis=0)

    return centroids, labels
