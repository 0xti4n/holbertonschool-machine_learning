#!/usr/bin/env python3
"""derivative of a polynomial"""


def poly_derivative(poly):
    """calculates the derivative
    of a polynomial
    """
    if len(poly) == 0:
        return None

    derivate = [poly[i] * i for i in range(1, len(poly))]

    if len(derivate) == 0:
        return [0]

    return derivate
