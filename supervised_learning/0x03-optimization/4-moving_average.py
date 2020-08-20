#!/usr/bin/env python3
"""Moving Average"""
import numpy as np


def moving_average(data, beta):
    """calculates the weighted
    moving average of a data set"""
    wma = []
    prev = 0
    for i in range(len(data)):
        v = prev * beta + (1 - beta) * data[i]
        wma.append(v / (1 - beta ** (i + 1)))
        prev = v
    return wma
