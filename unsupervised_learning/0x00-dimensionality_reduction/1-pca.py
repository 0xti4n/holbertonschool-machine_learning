#!/usr/bin/env python3
"""PCA v2"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset:

    -> X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point

    -> ndim is the new dimensionality of the transformed X

    -> Returns: T, a numpy.ndarray of shape (n, ndim)
        containing the transformed version of X
    """
    n, d = X.shape

    X_m = X - X.mean(axis=0)
    u, s, v = np.linalg.svd(X_m)
    W = v[:ndim]

    T = np.matmul(X_m, W.T)

    return T
