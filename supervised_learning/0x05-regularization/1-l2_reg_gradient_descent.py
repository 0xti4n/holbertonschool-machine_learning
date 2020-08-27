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
    dw = {}
    db = {}
    error = cache["A" + str(L)] - Y
    for i in reversed(range(1, L + 1)):
        A_p = cache["A" + str(i - 1)]
        A = cache["A" + str(i)]
        w = "W" + str(i)
        b = "b" + str(i)
        if i == L:
            dw[w] = np.matmul(error, A_p.T) / m + (lambtha * weights[w]) / m
            db[b] = np.sum(error, axis=1, keepdims=True) / m

        else:
            dw_nxt = "W" + str(i + 1)
            db_nxt = "b" + str(i + 1)
            w_p = weights["W" + str(i)]
            w_nxt = weights["W" + str(i + 1)]
            error = np.matmul(w_nxt.T, error) * (1 - (A * A))
            dw[w] = np.matmul(error, A_p.T) / m + (lambtha * w_p) / m
            db[b] = np.sum(error, axis=1, keepdims=True) / m

            weights[dw_nxt] -= alpha * dw[dw_nxt]
            weights[db_nxt] -= alpha * db[db_nxt]

            if i == 1:
                weights['W1'] -= alpha * dw[w]
                weights['b1'] -= alpha * db[b]
