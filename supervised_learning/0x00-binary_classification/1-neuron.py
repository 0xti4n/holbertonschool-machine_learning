#!/usr/bin/env python3
"""single neuron performing binary classification"""
import numpy as np


class Neuron():
    """class Neuron that
    defines a single neuron"""
    def __init__(self, nx):
        """constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive')
        self.__W = np.random.normal(0, 1, size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter of W"""
        return self.__W

    @property
    def b(self):
        """getter of b"""
        return self.__b

    @property
    def A(self):
        """getter of A"""
        return self.__A
