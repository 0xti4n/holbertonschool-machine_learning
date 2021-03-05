#!/usr/bin/env python3
"""
Simple Policy function
"""
import numpy as np


def policy(matrix, weight):
    """calculate simple policy

    Args:
    -> matrix that contains the states
    -> weight an array of weights

    return simple policy
    """
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp / exp.sum()


def policy_gradient(state, weight):
    """computes the Monte-Carlo policy
    gradient based on a state and a weight matrix

    Args:
    -> state: matrix representing the current
        observation of the environment
    -> weight: matrix of random weight

    Return: the action and the gradient (in this order)
    """
    prob = policy(state, weight)

    action = np.random.choice(len(prob[0]), p=prob[0])
    s = prob.reshape(-1, 1)

    softmax = np.diagflat(s) - np.dot(s, s.T)

    dsoftmax = softmax[action, :]

    dlog = dsoftmax / prob[0, action]

    grad = state.T.dot(dlog[None, :])

    return action, grad
