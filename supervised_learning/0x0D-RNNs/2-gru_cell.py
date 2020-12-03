#!/usr/bin/env python3
"""GRU Cell"""
import numpy as np


def softmax(Z):
    """softmax activation"""
    exps = np.exp(Z)
    return exps / np.sum(exps, axis=1, keepdims=True)


def sigmoid(X):
    """sigmoid Activation"""
    return 1.0 / (1.0 + np.exp(-X))


class GRUCell():
    """represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """
            -> i is the dimensionality of the data
            -> h is the dimensionality of the hidden state
            -> o is the dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step

        -> x_t is a numpy.ndarray of shape (m, i) that
            contains the data input for the cell
            * m is the batche size for the data

        -> h_prev is a numpy.ndarray of shape (m, h)
            containing the previous hidden state

        -> Returns: h_next, y
            * h_next is the next hidden state
            * y is the output of the cell
        """
        concat = np.hstack((h_prev, x_t))
        z = sigmoid(np.dot(concat, self.Wz) + self.bz)
        r = sigmoid(np.dot(concat, self.Wr) + self.br)

        concat_1 = np.hstack((r * h_prev, x_t))
        h_ = np.tanh(np.dot(concat_1, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_
        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
