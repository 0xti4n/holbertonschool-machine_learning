#!/usr/bin/env python3
"""transpose
    of a 2D matrix
"""


def matrix_transpose(matrix):
    """function that returns
        the transpose
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
