#!/usr/bin/env python3
"""Absorbing Markov Chain"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing:

    -> P is a is a square 2D numpy.ndarray of shape
        (n, n) representing the standard transition matrix
        * P[i, j] is the probability of transitioning from state i to state j
        * n is the number of states in the markov chain

    -> Returns: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False

    n, _ = P.shape
    n1 = n // 2
    if (P == np.eye(n)).all():
        return True

    if (np.diag(P) == 1).any():
        p = P[0:n1, 0:n1]
        if (p == np.eye(n1)).all():
            return True
        for i in range(n):
            for j in range(n):
                if i == j and i + 1 < n and j + 1 < n:
                    if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                        return False
        return True
    return False
