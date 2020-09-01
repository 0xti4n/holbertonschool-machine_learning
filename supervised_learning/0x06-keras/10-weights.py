#!/usr/bin/env python3
"""Save and Load Weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """save file"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Load file"""
    network.load_weights(filename)
    return None
