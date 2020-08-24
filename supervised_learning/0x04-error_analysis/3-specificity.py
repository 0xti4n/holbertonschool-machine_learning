#!/usr/bin/env python3
"""Specificity"""
import numpy as np


def specificity(confusion):
    """calculates the specificity
    for each class in a confusion matrix"""

    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    FN = confusion.sum(axis=1) - TP
    TN = confusion.sum() - (FP + FN + TP)

    return TN / (TN + FP)
