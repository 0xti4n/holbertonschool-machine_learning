#!/usr/bin/env python3
"""Likelihood Bayesian prob"""
import numpy as np


def likelihood(x, n, P):
    """calculates the likelihood of obtaining this
    data given various hypothetical probabilities
    of developing severe side effects:

    -> x is the number of patients that develop severe side effects

    -> n is the total number of patients observed

    -> P is a 1D numpy.ndarray containing the various
        hypothetical probabilities of developing severe side effects

    -> Returns: a 1D numpy.ndarray containing the likelihood
        of obtaining the data, x and n, for each
        probability in P, respectively
    """
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, (int, float)) or x < 0:
        msg = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(msg)

    if x > n:
        raise ValueError('x cannot be greater than n')

    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    msg = 'All values in P must be in the range [0, 1]'
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError(msg)

    f_n = np.math.factorial(n)
    f_x = np.math.factorial(x) * np.math.factorial(n - x)
    result = f_n / f_x
    p_pow = P ** x

    return result * p_pow * ((1 - P) ** (n - x))
