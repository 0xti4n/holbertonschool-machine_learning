#!/usr/bin/env python3
"""Save and Load Model"""
import tensorflow.keras as K


def save_model(network, filename):
    """ saves an entire mode"""
    network.save(filename)
    return None


def load_model(filename):
    """Load the entire model"""
    return K.models.load_model(filename)
