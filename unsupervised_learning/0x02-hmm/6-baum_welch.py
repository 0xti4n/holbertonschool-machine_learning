#!/usr/bin/env python3
"""Baum-Welch Algorithm Markov Chain"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model:

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
        * Transition[i, j] is the probability of
            transitioning from the hidden state i to j

    -> Initial a numpy.ndarray of shape (N, 1) containing
        the probability of starting in a particular hidden state

    -> Returns: P, F, or None, None on failure
        * P is the likelihood of the observations given the model
        * F is a numpy.ndarray of shape (N, T) containing
            the forward path probabilities
            ** F[i, j] is the probability of being in
                hidden state i at time j given the previous observations
    """

    T = Observation.shape[0]
    N, _ = Transition.shape
    F = np.zeros((N, T))

    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            mul_f_t = F[:, t - 1].dot(Transition[:, j])
            res = mul_f_t * Emission[j, Observation[t]]
            F[j, t] = np.sum(res)
    P = F[:, -1].sum()
    return P, F


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model:

    -> Observation is a numpy.ndarray of shape (T,) that
        contains the index of the observation

        * T is the number of observations

    -> Emission is a numpy.ndarray of shape (N, M) containing
        the emission probability of a specific observation given a hidden state

        * Emission[i, j] is the probability of observing
            j given the hidden state i
        * N is the number of hidden states
        * M is the number of all possible observations

    -> Transition is a 2D numpy.ndarray of shape (N, N)
        containing the transition probabilities
        * Transition[i, j] is the probability of transitioning
            from the hidden state i to j

    -> Initial a numpy.ndarray of shape (N, 1) containing
        the probability of starting in a particular hidden state

    -> Returns: P, B, or None, None on failure

        * P is the likelihood of the observations given the model
        * B is a numpy.ndarray of shape (N, T) containing
            the backward path probabilities

            ** B[i, j] is the probability of generating the
                future observations from hidden state i at time j
    """
    N, _ = Transition.shape
    T = Observation.shape[0]

    beta = np.zeros((N, T))

    beta[:, T - 1] = np.ones((N))

    for t in range(T - 2, -1, -1):
        for j in range(N):
            tmp = beta[:, t + 1] * Emission[:, Observation[t + 1]]
            tmp = tmp * Transition[j, :]
            beta[j, t] = np.sum(tmp)

    p = np.sum(Initial.T * Emission[:, Observation[0]] * beta[:, 0])

    return p, beta


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """performs the Baum-Welch algorithm for a hidden markov model:

    -> Observations is a numpy.ndarray of shape (T,) that contains
        the index of the observation

        * T is the number of observations

    -> Transition is a numpy.ndarray of shape (M, M) that
        contains the initialized transition probabilities

        * M is the number of hidden states

    -> Emission is a numpy.ndarray of shape (M, N) that contains
        the initialized emission probabilities

        * N is the number of output states

    -> Initial is a numpy.ndarray of shape (M, 1) that
        contains the initialized starting probabilities

    -> iterations is the number of times expectation-maximization
        should be performed

    -> Returns: the converged Transition,
        Emission, or None, None on failure
    """
    T = Observations.shape[0]
    M = Transition.shape[0]

    for n in range(1, iterations):
        _, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((M, M, T - 1))

        for t in range(T - 1):
            aux = np.dot(F[:, t].T, Transition)
            aux = aux * Emission[:, Observations[t + 1]].T

            den = np.dot(aux, B[:, t + 1])
            for i in range(M):
                aux1 = F[i, t] * Transition[i]
                aux1 = aux1 * Emission[:, Observations[t + 1]].T
                num = aux1 * B[:, t + 1].T
                xi[i, :, t] = num / den

        gamma = xi.sum(1)
        sum_gamma = np.sum(gamma, axis=1).reshape((-1, 1))
        Transition = np.sum(xi, axis=2) / sum_gamma

        gamma = np.hstack((gamma,
                           np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = Emission.shape[1]
        den = np.sum(gamma, axis=1)

        for l in range(K):
            Emission[:, l] = np.sum(gamma[:, Observations == l], axis=1)

        Emission = np.divide(Emission, den.reshape((-1, 1)))

    return Transition, Emission
