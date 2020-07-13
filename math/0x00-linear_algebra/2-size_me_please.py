#!/usr/bin/env python3
"""calculates
    the shape of a matrix
"""


def matrix_shape(matrix):
    """ function that calculates
        the shape of a matrix
    """
    shape_matrix = []
    len_matrix = len(matrix)

    shape_matrix.append(len_matrix)
    shape_matrix.append(len(matrix[0]))

    try:
        shape_matrix.append(len(matrix[0][0]))
    except TypeError:
        pass

    return shape_matrix
