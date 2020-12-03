#!/usr/bin/env python3
"""Bidirectional Cell Backward"""
import numpy as np


def softmax(Z):
    """softmax activation"""
    exps = np.exp(Z)
    return exps / np.sum(exps, axis=1, keepdims=True)


class BidirectionalCell():
    """represents a bidirectional
        cell of an RNN"""

    def __init__(self, i, h, o):
        """
            -> i is the dimensionality of the data
            -> h is the dimensionality of the hidden states
            -> o is the dimensionality of the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """calculates the hidden state in the forward
            direction for one time step

        -> x_t is a numpy.ndarray of shape (m, i) that
            contains the data input for the cell
            * m is the batch size for the data

        -> h_prev is a numpy.ndarray of shape (m, h)
            containing the previous hidden state

        -> Returns: h_next, the next hidden state
        """
        concat = np.hstack((h_prev, x_t))
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """calculates the hidden state in the backward
            direction for one time step

        -> x_t is a numpy.ndarray of shape (m, i)
            that contains the data input for the cell
            * m is the batch size for the data

        -> h_next is a numpy.ndarray of shape (m, h)
            containing the next hidden state

        -> Returns: h_pev, the previous hidden state
        """
        concat = np.hstack((h_next, x_t))
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """calculates all outputs for the RNN:

        -> H is a numpy.ndarray of shape (t, m, 2 * h) that
            contains the concatenated hidden states from
            both directions, excluding their initialized states
            * t is the number of time steps
            * m is the batch size for the data
            * h is the dimensionality of the hidden states

        -> Returns: Y, the outputs
        """
        t, m, h = H.shape
        Y = []

        for i in range(t):
            y = softmax(np.dot(H[i], self.Wy) + self.by)
            Y.append(y)

        return np.array(Y)
