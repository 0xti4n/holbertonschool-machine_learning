#!/usr/bin/env python3
"""Mean and Covariance"""
import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set:

    -> X is a numpy.ndarray of shape (n, d) containing the data set:
        * n is the number of data points
        * d is the number of dimensions in each data point
        * If X is not a 2D numpy.ndarray, raise a TypeError
            with the message X must be a 2D numpy.ndarray
        * If n is less than 2, raise a ValueError with the
            message X must contain multiple data points

    -> Returns: mean, cov:
        * mean is a numpy.ndarray of shape (1, d) containing
            the mean of the data set
        * cov is a numpy.ndarray of shape (d, d) containing
            the covariance matrix of the data set
    """

    if type(X) is not np.ndarray:
        raise TypeError('X must be a 2D numpy.ndarray')

    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')

    n, d = X.shape
    mean = X.mean(axis=0)
    cov = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov[i, j] += np.matmul(X[:, i] - mean[i], X[:, j] - mean[j])

    cov = cov / (n - 1)
    return mean, cov
