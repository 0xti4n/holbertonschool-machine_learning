#!/usr/bin/env python3
"""adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Function that adds two elements"""
    if len(arr1) != len(arr2):
        return None
    new_list = []

    for i in range(len(arr1)):
        add = arr1[i] + arr2[i]
        new_list.append(add)
    return new_list
