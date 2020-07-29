#!/usr/bin/env python3
"""poisson distribution"""


def fact(n):
    """Factorial"""
    if n <= 1:
        return 1
    return n * fact(n - 1)


class Poisson():
    """class Poisson that represents
    a poisson distribution
    """
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
            new_lambtha = 0
            for i in data:
                new_lambtha += i / len(data)
            self.lambtha = round(new_lambtha, 2)

    def pmf(self, k):
        """Calculates the value of the
        Probability Mass Function
        for a given number of successes"""
        if type(k) is not int:
            k = int(k)

        if k >= 0:
            PMF = self.lambtha ** k * 2.7182818285 ** -self.lambtha / fact(k)
            return PMF
        return 0

    def cdf(self, k):
        """Calculates the value of the
        Cumulative Distribution Function
        for a given number of successes"""
        if type(k) is not int:
            k = int(k)

        if k >= 0:
            CDF = 0
            for i in range(k + 1):
                power1 = self.lambtha ** i
                power2 = 2.7182818285 ** -self.lambtha
                CDF += power1 * power2 / fact(i)
            return CDF
        return 0
