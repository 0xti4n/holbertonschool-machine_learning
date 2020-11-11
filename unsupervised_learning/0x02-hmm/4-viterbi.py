#!/usr/bin/env python3
"""Viretbi Algorithm Markov Chain"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """calculates the most likely sequence of hidden
        states for a hidden markov model:

    -> Observation is a numpy.ndarray of shape (T,) that
        contains the index of the observation

        * T is the number of observations

    -> Emission is a numpy.ndarray of shape (N, M) containing
        the emission probability of a specific observation
        given a hidden state

        * Emission[i, j] is the probability of observing j
            given the hidden state i
        * N is the number of hidden states
        * M is the number of all possible observations

    -> Transition is a 2D numpy.ndarray of shape (N, N)
        containing the transition probabilities

        * Transition[i, j] is the probability of transitioning
            from the hidden state i to j

    -> Initial a numpy.ndarray of shape (N, 1) containing
        the probability of starting in a particular hidden state

    -> Returns: path, P, or None, None on failure

        * path is the a list of length T containing the most
            likely sequence of hidden states
        * P is the probability of obtaining the path sequence
    """
    N, _ = Transition.shape
    T = Observation.shape[0]
    path = []

    D = np.zeros((N, T))
    idx = np.zeros((N, T))

    D[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])

    for t in range(1, T):
        for j in range(N):
            mult_f_t = np.multiply(D[:, t - 1], Transition[:, j])
            prob = mult_f_t * Emission[j, Observation[t]]
            D[j, t] = np.max(prob)
            idx[j, t] = np.argmax(prob)

    s = np.argmax(D[:, -1])
    path.append(s)

    for n in range(T - 1, 0, -1):
        s = int(idx[s, n])
        path.append(s)

    p = np.max(D[:, -1])
    path = path[::-1]
    return path, p
