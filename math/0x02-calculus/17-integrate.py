#!/usr/bin/env python3
"""integral of a polynomial"""


def poly_integral(poly, C=0):
    """
    function that calculates
    the integral of a polynomial:
    """
    if type(poly) != list or len(poly) == 0:
        return None
    if type(C) != int:
        return None

    coefficient = []
    coefficient.append(C)

    for i in poly[::-1]:
        if i == 0:
            poly.pop(i)
        else:
            break

    for idx, num in zip(range(1, len(poly) + 1), poly):
        integral = num / idx
        if type(num) is not int and type(num) is not float:
            return None
        if int(integral) == integral:
            coefficient.append(int(integral))
        else:
            coefficient.append(integral)
    return coefficient
