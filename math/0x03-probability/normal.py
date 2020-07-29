#!/usr/bin/env python3
"""normal distribution"""


class Normal():
    """class Normal that represents
    a normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if self.stddev < 0:
                raise ValueError('stddev must be a positive value')
        if data:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            result = sum(data)
            self.mean = result / len(data)
            variance = 0
            for i in data:
                variance += (i - self.mean) ** 2
            variance = variance / len(data)
            self.stddev = variance ** (1.0 / 2)

    def z_score(self, x):
        """Calculates the z-score
        of a given x-value"""
        Z = (x - self.mean) / self.stddev
        return Z

    def x_value(self, z):
        """Calculates the x-value
        of a given z-score"""
        X = self.stddev * z + self.mean
        return X

    def pdf(self, x):
        """Calculates the value of
        the Probability Density Function
        for a given x-value"""
        r1 = 1.0 / (self.stddev * (2.0 * 3.1415926536) ** (1/2))
        r2 = 2.7182818285 ** (-1.0 * (x - self.mean) ** 2 / (2.0 * (
            self.stddev ** 2)))
        PDF = r1 * r2
        return PDF

    def cdf(self, x):
        """Calculates the value of
        the Cumulative Distribution Function
        for a given x-value"""
        x = (x - self.mean) / (self.stddev * (2.0 ** (1.0/2)))
        erf = (2.0 / (3.1415926536**(1.0/2))) * (x - (x**3)/3 + (x**5)/10 -
                                                 (x**7)/42 + (x**9)/216)

        return 0.5 * (1.0 + erf)
