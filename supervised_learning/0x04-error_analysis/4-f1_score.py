#!/usr/bin/env python3
"""  F1 score"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score
    of a confusion matrix"""

    recall = sensitivity(confusion)
    prec = precision(confusion)

    adds = prec + recall
    mul = prec * recall

    return 2 * mul / adds
