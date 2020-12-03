#!/usr/bin/env python3
"""LSTM Cell"""
import numpy as np


def softmax(Z):
    """softmax activation"""
    exps = np.exp(Z)
    return exps / np.sum(exps, axis=1, keepdims=True)


def sigmoid(X):
    """sigmoid Activation"""
    return 1.0 / (1.0 + np.exp(-X))


class LSTMCell():
    """represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
            -> i is the dimensionality of the data
            -> h is the dimensionality of the hidden state
            -> o is the dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """performs forward propagation for one time step

        -> x_t is a numpy.ndarray of shape (m, i) that
            contains the data input for the cell
            * m is the batche size for the data

        -> h_prev is a numpy.ndarray of shape (m, h)
            containing the previous hidden state

        -> c_prev is a numpy.ndarray of shape (m, h)
            containing the previous cell state

        -> Returns: h_next, c_next, y
            * h_next is the next hidden state
            * c_next is the next cell state
            * y is the output of the cell
        """
        concat = np.hstack((h_prev, x_t))
        ft = sigmoid(np.dot(concat, self.Wf) + self.bf)
        ut = sigmoid(np.dot(concat, self.Wu) + self.bu)

        cct = np.tanh(np.dot(concat, self.Wc) + self.bc)
        c_next = np.multiply(ft, c_prev) + np.multiply(ut, cct)

        ot = sigmoid(np.dot(concat, self.Wo) + self.bo)

        h_next = np.multiply(ot, np.tanh(c_next))
        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
