#!/usr/bin/env python3
"""Summation"""


def summation_i_squared(n):
    """Calculates summation"""

    if n == 0:
        return 0
    return n ** 2 + summation_i_squared(n - 1)

