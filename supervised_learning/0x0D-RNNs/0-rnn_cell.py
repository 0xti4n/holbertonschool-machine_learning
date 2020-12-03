#!/usr/bin/env python3
"""RNN Cell"""
import numpy as np


def softmax(Z):
    """softmax activation"""
    exps = np.exp(Z)
    return exps / np.sum(exps, axis=1, keepdims=True)


class RNNCell():
    """represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """
            -> i is the dimensionality of the data
            -> h is the dimensionality of the hidden state
            -> o is the dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step

        -> x_t is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
            * m is the batche size for the data

        -> h_prev is a numpy.ndarray of shape (m, h) containing
            the previous hidden state

        -> Returns: h_next, y
            * h_next is the next hidden state
            * y is the output of the cell
        """
        a = np.dot(np.hstack((h_prev, x_t)), self.Wh)
        h_next = np.tanh(a + self.bh)
        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
