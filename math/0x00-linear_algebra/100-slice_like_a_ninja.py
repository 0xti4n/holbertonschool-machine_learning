#!/usr/bin/env python3
"""slices a matrix along a specific axes"""
import numpy as np


def np_slice(matrix, axes={}):
    """function that slices a matrix"""
    new_list = []
    keys = axes.keys()
    values = [i for i in range(max(keys) + 1)]
    for j in values:
        if j in keys:
            sl = slice((*axes[j]))
        else:
            sl = slice(None)
        new_list.append(sl)
    return matrix[tuple(new_list)]
