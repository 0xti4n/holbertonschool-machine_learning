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

    def pdf(self, x):
        """calculates the PDF at a data point:

        -> x is a numpy.ndarray of shape (d, 1) containing
        the data point whose PDF should be calculated
            * d is the number of dimensions of the Multinomial instance

        -> If x is not a numpy.ndarray, raise a TypeError
        with the message x must be a numpy.ndarray

        -> If x is not of shape (d, 1), raise a
        ValueError with the message x must have the shape ({d}, 1)

        -> Returns the value of the PDF
        """

        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')

        shape = (self.cov.shape[0], 1)
        if len(x.shape) != 2 or x.shape != shape:
            msg = 'x must have the shape ({}, 1)'.format(shape[0])
            raise ValueError(msg)

        N = x.size

        tmp1 = np.linalg.det(self.cov) ** (-1/2)
        mul1 = -.5 * (x - self.mean).T
        tmp2 = np.exp(mul1 @ np.linalg.inv(self.cov) @ (x - self.mean))

        res = (2 * np.pi) ** (-N/2) * tmp1 * tmp2

        return np.squeeze(res)
