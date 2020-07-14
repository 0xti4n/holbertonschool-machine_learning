#!/usr/bin/env python3
"""transposes matrix"""


def np_transpose(matrix):
    """Function that transposes matrix"""
    new_matrix = matrix[:]

    return new_matrix.transpose()
