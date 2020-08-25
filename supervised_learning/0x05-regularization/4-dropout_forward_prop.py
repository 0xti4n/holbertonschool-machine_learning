#!/usr/bin/env python3
"""Forward Propagation with Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout

    -> X is a numpy.ndarray of shape (nx, m) containing the
        input data for the network
        * nx is the number of input features
        * m is the number of data points
    -> weights is a dictionary of the weights and biases
        of the neural network
    -> L the number of layers in the network
    -> keep_prob is the probability that a node will be kept
    -> All layers except the last should use the tanh activation function
    -> The last layer should use the softmax activation function
    -> Returns: a dictionary containing the outputs of each
        layer and the dropout mask used on each layer (see example for format)
    """
    cache = {}
    cache["A0"] = X
    for lidx in range(1, L + 1):
        A_prev = cache["A" + str(lidx - 1)]

        W = weights['W' + str(lidx)]
        b = weights['b' + str(lidx)]

        Z = np.matmul(W, A_prev) + b

        if lidx != L:
            A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            cache['D' + str(lidx)] = D
            A = A * D
            A = A / keep_prob
            cache['A' + str(lidx)] = A

        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
            cache['A' + str(lidx)] = A

    return cache
