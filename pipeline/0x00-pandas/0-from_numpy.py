#!/usr/bin/env python3
"""creates a pd.DataFrame from a np.ndarray"""
import pandas as pd


def from_numpy(array):
    """creates a pd.DataFrame from a np.ndarray:

    Args:
    -> array is the np.ndarray from which you should
    create the pd.DataFrame

    Returns: the newly created pd.DataFrame
    """
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
              'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    size = array.shape[1]

    if size > 26:
        data = pd.DataFrame(array)
        data = data.iloc[:, 0:26]
        data.columns = [i.upper() for i in labels]

    else:
        col = [labels[i].upper() for i in range(size)]
        data = pd.DataFrame(array, columns=col)

    return data
