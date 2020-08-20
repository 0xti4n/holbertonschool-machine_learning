#!/usr/bin/env python3
"""Normalization"""
import numpy as np


def normalize(X, m, s):
    """normalizes (standardizes)
    a matrix"""
    X -= m
    X /= s
    return X
