#!/usr/bin/env python3
"""Shuffle Data"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points
    in two matrices the same way"""
    n = X.shape[0]
    shuffle = np.random.permutation(n)
    return X[shuffle], Y[shuffle]
