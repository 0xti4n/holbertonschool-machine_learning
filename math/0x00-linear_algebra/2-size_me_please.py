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

    while type(matrix[0]) == list:
        matrix = matrix[0]
        shape_matrix.append(len(matrix))

    return shape_matrix
