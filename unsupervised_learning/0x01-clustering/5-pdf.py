#!/usr/bin/env python3
"""PDF Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """calculates the probability density
        function of a Gaussian distribution:

    -> X is a numpy.ndarray of shape (n, d) containing
        the data points whose PDF should be evaluated

    -> m is a numpy.ndarray of shape (d,) containing
        the mean of the distribution

    -> S is a numpy.ndarray of shape (d, d) containing
        the covariance of the distribution

    -> Returns: P, or None on failure
        * P is a numpy.ndarray of shape (n,) containing
            the PDF values for each data point
        * All values in P have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None

    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    if S.shape[0] != S.shape[1]:
        return None

    if X.shape[1] != m.shape[0]:
        return None

    """https://scipython.com/blog/
        visualizing-the-bivariate-gaussian-distribution/"""
    """https://pythonrobotics.readthedocs.io/
        en/latest/modules/appendix.html"""

    d = m.shape[0]

    X_m = X - m
    s_det = np.linalg.det(S)
    s_inv = np.linalg.inv(S)
    N = np.sqrt((2 * np.pi) ** d * s_det)
    fac = np.einsum('...k,kl,...l->...', X_m, s_inv, X_m)

    p = (1. / N) * np.exp(-fac / 2)

    return np.maximum(p, 1e-300)
