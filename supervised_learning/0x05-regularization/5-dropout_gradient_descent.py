#!/usr/bin/env python3
"""Gradient Descent with Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of a neural network with Dropout regularization
    using gradient descent

    -> Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data
            * classes is the number of classes
            * m is the number of data points
    -> weights is a dictionary of the weights and biases of the neural network
    -> cache is a dictionary of the outputs and dropout masks of each layer
        of the neural network
    -> alpha is the learning rate
    -> keep_prob is the probability that a node will be kept
    -> L is the number of layers of the network
    -> All layers use the tanh activation function except the last, which
        uses the softmax activation function
    -> The weights of the network should be updated in place
    """
    m = Y.shape[1]

    error = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        w = "W" + str(i)
        A_p = cache["A" + str(i - 1)]
        if i == L:
            dw = (1 / m) * np.matmul(error, A_p.T) / m
        else:
            error = np.matmul(weights["W" + str(i + 1)].T, error)
            da2 = error * cache["D" + str(i)]
            da2 /= keep_prob
            dz2 = da2 * (1 - (cache["A" + str(i)] * cache["A" + str(i)]))
            dw = np.matmul(dz2, A_p.T) / m
            weights[w] -= alpha * dw
