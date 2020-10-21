#!/usr/bin/env python3
"""Minor"""


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


def find_minor(m, i, j):
    """find the minor of the matrix"""
    new_matrix = [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]
    det = determinant(new_matrix)
    return det


def minor(matrix):
    """calculates the minor matrix of a matrix:

    -> matrix is a list of lists whose minor matrix
        should be calculated

    -> If matrix is not a list of lists,
        raise a TypeError with the message matrix must be a list of lists

    -> If matrix is not square or is empty, raise a
        ValueError with the message matrix must be a non-empty square matrix

    -> Returns: the minor matrix of matrix
    """
    A = matrix
    n = len(A)
    new = [cpy[:] for cpy in A]

    if type(A) is not list or len(A) == 0:
        raise TypeError('matrix must be a list of lists')

    if len(A) != len(A[0]) or len(A[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')

    if len(A) == 1 and len(A[0]) == 1:
        return [[1]]

    for idx in range(n):
        for j in range(n):
            new[idx][j] = find_minor(A, idx, j)

    return new
