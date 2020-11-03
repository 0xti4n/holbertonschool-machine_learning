#!/usr/bin/env python3
"""Variance K-means"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster
        variance for a data set:

    -> X is a numpy.ndarray of shape (n, d)
        containing the data set

    -> C is a numpy.ndarray of shape (k, d)
        containing the centroid means for each cluster

    -> Returns: var, or None on failure
        * var is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    try:
        variance = np.sum((X - C[:, np.newaxis]) ** 2, -1)
        D = np.sqrt(variance)
        D = np.min(D, axis=0)

        var = np.sum(D ** 2)

        return var
    except ValueError:
        return None
