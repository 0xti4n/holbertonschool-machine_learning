#!/usr/bin/env python3
"""deep neural network"""
import numpy as np


def sigmoid(X):
    """sigmoid Activation"""
    return 1.0 / (1.0 + np.exp(-X))


def linear_formula(A, W, b):
    """Linear formula"""
    Z = np.matmul(W, A) + b
    return Z


def activation(A_prev, W, b):
    """Activation function"""
    Z = linear_formula(A_prev, W, b)
    A = sigmoid(Z)
    return A


class DeepNeuralNetwork():
    """deep neural network performing
    binary classification"""
    def __init__(self, nx, layers):
        """constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(layers) is not list:
            raise TypeError('layers must be a list of positive integers')

        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lidx in range(self.__L):
            if type(layers[lidx]) is not int or layers[lidx] < 1:
                raise TypeError('layers must be a list of positive integers')

            self.__weights['b' + str(lidx+1)] = np.zeros((layers[lidx], 1))

            if lidx == 0:
                sqr = np.sqrt(2 / nx)
                formula = np.random.randn(layers[lidx], nx) * sqr
                self.__weights['W' + str(lidx+1)] = formula
            else:
                sqr = np.sqrt(2 / layers[lidx - 1])
                formula = np.random.randn(layers[lidx], layers[lidx - 1]) * sqr
                self.__weights['W' + str(lidx+1)] = formula

    @property
    def L(self):
        """getter of L"""
        return self.__L

    @property
    def cache(self):
        """getter of cache"""
        return self.__cache

    @property
    def weights(self):
        """getter of weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation
        of the neural network"""
        A = X
        for lidx in range(1, self.__L + 1):
            A_prev = A

            W = self.__weights['W' + str(lidx)]
            b = self.__weights['b' + str(lidx)]

            A = activation(A_prev, W, b)
            self.__cache['A' + str(lidx)] = A

        return A, self.__cache
