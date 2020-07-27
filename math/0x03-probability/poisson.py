#!/usr/bin/env python3
"""poisson distribution"""


class Poisson():
    """class Poisson that represents
    a poisson distribution
    """
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha < 0:
                raise ValueError('lambtha must be a positive value')
        if data:
            if type(data) is not list:
                raise ValueError('data must be a list')
            if len(data) < 2:
                raise('data must contain multiple values')
            new_lambtha = 0
            for i in data:
                new_lambtha += i / len(data)
            self.lambtha = round(new_lambtha, 2)
