#!/usr/bin/env python3
"""Determinant"""


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

    n = len(A)
    A_copy = A.copy()

    for idx in range(n):
        if A_copy[idx][idx] == 0:
            A_copy[idx][idx] = 0
        for i in range(idx + 1, n):
            number = A_copy[i][idx] / A_copy[idx][idx]
            for j in range(n):
                A_copy[i][j] = A_copy[i][j] - number * A_copy[idx][j]

    product = 1
    for i in range(n):
        product *= A_copy[i][i]

    return round(product)
