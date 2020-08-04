#!/usr/bin/env python3
"""deep neural network"""
import numpy as np


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
            self.__weights['b' + str(lidx+1)] = np.zeros((layers[lidx], 1))
            if type(layers[lidx]) is not int or layers[lidx] < 1:
                raise TypeError('layers must be a list of positive integers')
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
