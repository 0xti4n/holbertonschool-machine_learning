#!/usr/bin/env python3
"""Determinant"""


def Minor(A, i, j):
    """MINOR"""
    return [row[:j] + row[j+1:] for row in (A[:i]+A[i+1:])]


def determinant(matrix):
    """calculates the determinant of a matrix:

    -> matrix is a list of lists whose determinant
    should be calculated

    -> If matrix is not a list of lists, raise a
    TypeError with the message matrix must be a list of lists

    -> If matrix is not square, raise a ValueError
    with the message matrix must be a square matrix

    -> The list [[]] represents a 0x0 matrix

    -> Returns: the determinant of matrix
    """
    A = matrix

    if type(A) is not list or len(A) == 0:
        raise TypeError('matrix must be a list of lists')

    if all([type(i) is list for i in A]) is False:
        raise TypeError('matrix must be a list of lists')

    if A[0] and len(A) != len(A[0]):
        raise ValueError('matrix must be a square matrix')

    if A == [[]]:
        return 1

    if len(A) == 1:
        return A[0][0]

    if len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]

    n = len(A)

    det = 0
    for c in range(n):
        det += ((-1)**c)*A[0][c]*determinant(Minor(A, 0, c))
    return det
