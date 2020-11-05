#!/usr/bin/env python3
""" Maximization EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """that calculates the maximization
        step in the EM algorithm for a GMM:

    -> X is a numpy.ndarray of shape (n, d)
        containing the data set

    -> g is a numpy.ndarray of shape (k, n)
        containing the posterior probabilities
        for each data point in each cluster

    -> Returns: pi, m, S, or None, None, None on failure
        * pi is a numpy.ndarray of shape (k,) containing
            the updated priors for each cluster
        * m is a numpy.ndarray of shape (k, d) containing
            the updated centroid means for each cluster
        * S is a numpy.ndarray of shape (k, d, d) containing
            the updated covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None

    n, d = X.shape

    pi = np.einsum('kn->k', g) / n
    m = np.einsum('kn,np->kp', g, X) / g.sum(1)[:, None]
    X_m = X-m[:, None, :]
    S = np.einsum('kn,knp,knq->kpq', g, X_m, X_m) / g.sum(1)[:, None, None]

    return pi, m, S
