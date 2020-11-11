#!/usr/bin/env python3
"""Regular Markov Chain"""
import numpy as np


def regular(P):
    """determines the steady state
    probabilities of a regular markov chain:

    -> P is a is a square 2D numpy.ndarray of shape
        (n, n) representing the transition matrix
        * P[i, j] is the probability of transitioning
            from state i to state j
        * n is the number of states in the markov chain

    -> Returns: a numpy.ndarray of shape (1, n)
    containing the steady state probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None

    if (P.dot(P) <= 0).any():
        return None

    if np.isclose(P.sum(1), 1) is False:
        return None

    n, _ = P.shape

    vals, vecs = np.linalg.eig(P.T)
    vecs1 = vecs[:, np.isclose(vals, 1)]
    vecs1 = vecs1[:, 0]

    s = vecs1 / vecs1.sum()

    return s.reshape((1, n))
