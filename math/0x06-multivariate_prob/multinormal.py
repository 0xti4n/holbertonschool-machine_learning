#!/usr/bin/env python3
"""class MultiNormal"""
import numpy as np


class MultiNormal():
    """Multinormal represents a
    Multivariate Normal distribution"""

    def __init__(self, data):
        """Initializer"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        data = data.T
        n, d = data.shape

        mean = np.expand_dims(data.mean(axis=0), axis=0)
        data = data - mean
        cov = data.T.dot(data) / (n - 1)
        self.mean = mean.reshape(d, 1)
        self.cov = cov
