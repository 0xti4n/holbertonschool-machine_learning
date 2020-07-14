#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """function that concatenates
        two matrices along a
        specific axis
    """
    if len(mat1) == 0 and len(mat2) == 0:
        return None

    new_list = [cpy[:] for cpy in mat1]
    new_list2 = [cpy1[:] for cpy1 in mat2]

    if axis == 0:
        result = new_list + new_list2

    elif axis == 1:
        new_matrix = []
        for i in range(len(new_list)):
            new_matrix.append(new_list[i] + new_list2[i])
        result = new_matrix
    return result
