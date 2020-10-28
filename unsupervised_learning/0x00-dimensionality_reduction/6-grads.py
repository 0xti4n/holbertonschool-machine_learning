#!/usr/bin/env python3
"""T-SNE Gradients"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """calculates the gradients of Y:

    -> Y is a numpy.ndarray of shape (n, ndim) containing
        the low dimensional transformation of X

    -> P is a numpy.ndarray of shape (n, n) containing
        the P affinities of X

    -> Returns: (dY, Q)
        * dY is a numpy.ndarray of shape (n, ndim)
            containing the gradients of Y
        * Q is a numpy.ndarray of shape (n, n)
            containing the Q affinities of Y
    """
    Q, num = Q_affinities(Y)
    n, ndim = Y.shape
    dy = np.zeros((n, ndim))

    pq_diff = P - Q
    for i in range(n):
        dy[i, :] = np.sum(np.tile(pq_diff[:, i] * num[:, i],
                                  (ndim, 1)).T * (Y[i, :] - Y), 0)

    return dy, Q
