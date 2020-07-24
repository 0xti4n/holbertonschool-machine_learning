#!/usr/bin/env python3
"""Summation"""


def summation_i_squared(n):
    """Calculates summation"""

    if n == 0:
        return 0
    res = n * n + summation_i_squared(n - 1)
    if not res:
        return None
    return res
