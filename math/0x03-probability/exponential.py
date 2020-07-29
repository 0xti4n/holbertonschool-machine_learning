#!/usr/bin/env python3
"""exponential distribution"""


class Exponential():
    """represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)
        if data is None:
            if self.lambtha < 0:
                raise ValueError('lambtha must be a positive value')
        if data:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            result = sum(data) / len(data)
            new_lambtha = 1 / result
            self.lambtha = new_lambtha

    def pdf(self, x):
        """Calculates the value of the
        Probability Density Function
        for a given time period"""
        if x >= 0:
            PDF = self.lambtha * 2.7182818285 ** -self.lambtha * x
            return PDF
        return 0

    def cdf(self, x):
        """Calculates the value of the
        Cumulative Distribution Function
        for a given time period"""
        if x >= 0:
            CDF = 1 - 2.7182818285 ** -self.lambtha * x
            return CDF
        return 0
