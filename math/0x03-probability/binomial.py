#!/usr/bin/env python3
"""binomial distribution"""


class Binomial():
    """class that represents a
    binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """constructor"""
        if data is not None:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = 0
            for i in data:
                variance += (i - mean) ** 2
            variance = variance / len(data)
            q = variance / mean
            p = 1 - q
            self.n = int(round(mean / p))
            self.p = float(mean / self.n)

        else:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
