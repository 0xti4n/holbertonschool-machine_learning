#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Function that concatenate two matrices"""
    new_matrix = np.concatenate((mat1, mat2), axis)

    return new_matrix
