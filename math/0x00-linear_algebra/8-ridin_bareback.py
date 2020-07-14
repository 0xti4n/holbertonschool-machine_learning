#!/usr/bin/env python3
""" performs matrix multiplication"""


def transpose(matrix):
    """function transpose
        of a 2D matrix
    """
    new_matrix2 = []
    len_matrix = len(matrix[0])

    for con in range(len_matrix):
        new_matrix = []
        for i in matrix:
            new_matrix.append(i[con])

        new_matrix2.append(new_matrix)
    return new_matrix2


def mat_mul(mat1, mat2):
    """Function that performs
    matrix multiplication"""

    if len(mat1[0]) != len(mat2):
        return None

    new_matrix2 = []
    mt3 = transpose(mat2)

    for i in mat1:
        new_matrix = []
        for j in mt3:
            mult = 0
            for l in range(len(j)):
                mult += i[l] * j[l]
            new_matrix.append(mult)
        new_matrix2.append(new_matrix)
    return new_matrix2
