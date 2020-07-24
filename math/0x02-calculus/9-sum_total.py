#!/usr/bin/env python3
"""Summation"""


def summation_i_squared(n):
    """Calculates summation"""

    if type(n) != int:
        return None

    res = n * (n + 1) * (2 * n + 1) // 6

    return res
