#!/usr/bin/env python3
"""adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Function that adds two matrices"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    new_matrix2 = []

    for i in range(len(mat1)):
        new_matrix = []
        for j in range(len(mat1[i])):
            add = mat1[i][j] + mat2[i][j]
            new_matrix.append(add)
        new_matrix2.append(new_matrix)
    return new_matrix2
