#!/usr/bin/env python3
"""Test"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """predict a neural network"""
    return network.predict(data, verbose=verbose)
