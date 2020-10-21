#!/usr/bin/env python3
"""Inverse"""


def transpose(matrix):
    """Transpose matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


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

    return new_matrix


def inverse(matrix):
    """calculates the inverse of a matrix:

    -> matrix is a list of lists whose inverse should be calculated

    -> If matrix is not a list of lists, raise a TypeError
        with the message matrix must be a list of lists

    -> If matrix is not square or is empty, raise a
        ValueError with the message matrix must be a non-empty square matrix

    Returns: the inverse of matrix, or None if matrix is singular"""
    A = matrix

    if type(A) is not list or len(A) == 0:
        raise TypeError('matrix must be a list of lists')

    if all([type(i) is list for i in A]) is False:
        raise TypeError('matrix must be a list of lists')

    if len(A) != len(A[0]) or len(A[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')

    if any([len(i) != len(A) for i in matrix]):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(A) == 1 and len(A[0]) == 1:
        return [[1]]

    n = len(A)
    new = [cpy[:] for cpy in A]

    for idx in range(n):
        for j in range(n):
            minor = find_minor(A, idx, j)
            new[idx][j] = (((-1)**(idx+j)) * determinant(minor))

    find_inverse = transpose(new)
    det = determinant(A)

    if det == 0:
        return None

    for row in range(len(find_inverse)):
        for col in range(len(find_inverse)):
            find_inverse[row][col] = find_inverse[row][col] / det

    return find_inverse
