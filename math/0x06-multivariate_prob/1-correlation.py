#!/usr/bin/env python3
"""Correlation"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix:

    -> C is a numpy.ndarray of shape (d, d)
        containing a covariance matrix
        * d is the number of dimensions
        * If C is not a numpy.ndarray, raise a TypeError
            with the message C must be a numpy.ndarray
        * If C does not have shape (d, d), raise a
            ValueError with the message C must be a 2D square matrix

    -> Returns a numpy.ndarray of shape (d, d)
        containing the correlation matrix
    """
    if type(C) is not np.ndarray or len(C.shape) != 2:
        raise TypeError('C must be a numpy.ndarray')

    if C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    v = np.sqrt(np.diag(C))
    out = np.outer(v, v)
    correlation = C / out

    return correlation
