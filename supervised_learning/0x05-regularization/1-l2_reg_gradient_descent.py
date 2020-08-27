#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    - Y is a one-hot numpy.ndarray of shape (classes, m) that
        contains the correct labels for the data
        * classes is the number of classes
        * m is the number of data points
    - weights is a dictionary of the weights and biases of
        the neural network
    - cache is a dictionary of the outputs of each layer of
        the neural network
    - alpha is the learning rate
    - lambtha is the L2 regularization parameter
    - L is the number of layers of the network
    - The neural network uses tanh activations on each layer except
        the last, which uses a softmax activation
    - The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]

    error = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        w = "W" + str(i)
        b = "b" + str(i)
        A_p = cache["A" + str(i - 1)]
        A = cache["A" + str(i)]
        if i == L:
            dw = (1 / m) * np.matmul(error, A_p.T) + lambtha / m * weights[w]
            db = np.sum(error, axis=1, keepdims=True) / m
        else:
            w1 = weights["W" + str(i + 1)].T
            error = np.matmul(w1, error) * (A * (1 - A))
            dw = np.matmul(error, A_p.T) + lambtha / m * weights[w]
            db = np.sum(error, axis=1, keepdims=True) / m

            weights[w] = weights[w] - alpha * dw
            weights[b] = weights[b] - alpha * db
