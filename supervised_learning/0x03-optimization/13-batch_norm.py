#!/usr/bin/env python3
""" Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of
    a neural network using batch normalization
    """
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    z_n = (Z - mean) / np.sqrt(variance + epsilon)
    z = gamma * z_n + beta
    return z
