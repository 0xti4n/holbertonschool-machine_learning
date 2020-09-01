#!/usr/bin/env python3
"""Test network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """tests a neural network"""
    return network.evaluate(data, labels, verbose=verbose)
