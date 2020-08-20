#!/usr/bin/env python3
"""Momentum """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates a variable using the gradient descent with
    momentum optimization algorithm"""
    vdw = (beta1 * v) + (1 - beta1) * grad
    var = var - (alpha * vdw)
    return var, vdw
