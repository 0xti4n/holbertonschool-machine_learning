#!/usr/bin/env python3
"""Epsilon Greedy"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """uses epsilon-greedy to determine the
    next action

    Args:
    -> Q is a numpy.ndarray containing the q-table
    -> state is the current state
    -> epsilon is the epsilon to use for the calculation

    Returns: the next action index
    """
    p = np.random.uniform(low=0.0, high=1.0)
    if p < epsilon:
        A = np.random.randint(low=0.0, high=len(Q[state]))
    else:
        A = np.argmax(Q[state])

    return A
