#!/usr/bin/env python3
"""derivative of a polynomial"""


def poly_derivative(poly):
    """calculates the derivative
    of a polynomial
    """
    if type(poly) != list or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivate = [poly[i] * i for i in range(1, len(poly))]

    return derivate
