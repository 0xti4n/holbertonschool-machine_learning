#!/usr/bin/env python3
"""performs element-wise addition,
    subtraction, multiplication,
    and division
"""


def np_elementwise(mat1, mat2):
    """function that performs element-wise"""
    new_list = []
    new_list.append(mat1 + mat2)
    new_list.append(mat1 - mat2)
    new_list.append(mat1 * mat2)
    new_list.append(mat1 / mat2)

    return tuple(new_list)
