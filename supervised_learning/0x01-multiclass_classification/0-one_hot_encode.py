#!/usr/bin/env python3
"""One-Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """function that converts a numeric
    label vector into a one-hot matrix"""

    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= np.max(Y):
        return None

    shape = (Y.size, Y.max() + 1)
    row = np.arange(shape[0])
    one = np.zeros((classes, shape[0]))
    one[Y, row] = 1
    return one
