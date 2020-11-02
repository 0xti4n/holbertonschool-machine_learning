#!/usr/bin/env python3
"""Initialize K-means"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means:

    -> X is a numpy.ndarray of shape (n, d) containing the dataset that
        will be used for K-means clustering
        * n is the number of data points
        * d is the number of dimensions for each data point

    -> k is a positive integer containing the number of clusters

    -> The cluster centroids should be initialized with a multivariate
        uniform distribution along each dimension in d:
        * The minimum values for the distribution should be the
            minimum values of X along each dimension in d
        * The maximum values for the distribution should be the
            maximum values of X along each dimension in d

    -> Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)

    centroids = np.random.uniform(low=0., high=1., size=(k, d))
    centroids = centroids * (x_max - x_min) + x_min

    return centroids
