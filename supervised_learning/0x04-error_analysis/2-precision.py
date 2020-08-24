#!/usr/bin/env python3
"""Precision"""
import numpy as np


def precision(confusion):
    """calculates the precision
    for each class in a confusion matrix"""

    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    return TP / (TP + FP)
