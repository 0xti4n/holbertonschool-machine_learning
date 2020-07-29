#!/usr/bin/env python3
"""binomial distribution"""


def fact(n):
    """Factorial"""
    if n <= 1:
        return 1
    return n * fact(n - 1)


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

    def pmf(self, k):
        """function Calculates the value of
        the Probability Mass Function for
        a given number of successes"""
        if type(k) is not int:
            k = int(k)

        if k >= 0:
            fact_n = fact(self.n)
            fact_k = fact(k)
            fact_n_k = fact(self.n - k)
            result = fact_k * fact_n_k
            combinatorie = fact_n / result
            PMF = combinatorie * self.p ** k * (1 - self.p) ** (self.n - k)
            return PMF
        return 0
