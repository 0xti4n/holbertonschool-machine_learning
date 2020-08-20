#!/usr/bin/env python3
"""RMSProp """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp
    optimization algorithm"""
    sdw = beta2 * s + ((1 - beta2) * (grad ** 2))
    w = var - alpha * grad / (np.sqrt(sdw) + epsilon)
    return w, sdw
