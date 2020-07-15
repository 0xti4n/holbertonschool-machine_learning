#!/usr/bin/env python3
"""calculates
    the shape of a matrix
"""


def matrix_shape(matrix):
    """ function that calculates
        the shape of a matrix
    """
    shape_matrix = []

    shape_matrix.append(len(matrix))
    shape_matrix.append(len(matrix[0]))

    if type(matrix[0][0]) != int:
        shape_matrix.append(len(matrix[0][0]))

    return shape_matrix
