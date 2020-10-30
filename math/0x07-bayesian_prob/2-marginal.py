#!/usr/bin/env python3
"""Marginal Probability Bayesian prob"""
import numpy as np


def marginal(x, n, P, Pr):
    """calculates the marginal probability
        of obtaining the data:

    -> x is the number of patients that
        develop severe side effects

    -> n is the total number of patients observed

    -> P is a 1D numpy.ndarray containing the
        various hypothetical probabilities of
        patients developing severe side effects

    -> Pr is a 1D numpy.ndarray containing
        the prior beliefs about P

    Returns: the marginal probability of obtaining x and n
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

    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        msg = 'Pr must be a numpy.ndarray with the same shape as P'
        raise TypeError(msg)

    msg = 'All values in P must be in the range [0, 1]'
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError(msg)

    msg = 'All values in Pr must be in the range [0, 1]'
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError(msg)

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    f_n = np.math.factorial(n)
    f_x = np.math.factorial(x) * np.math.factorial(n - x)
    result = f_n / f_x
    p_pow = P ** x

    inter = result * p_pow * ((1 - P) ** (n - x)) * Pr

    marginal = np.sum(inter)

    return marginal
