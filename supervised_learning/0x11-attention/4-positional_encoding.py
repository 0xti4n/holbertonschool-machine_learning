#!/usr/bin/env python3
"""Positional Encoding"""
import numpy as np


def get_angles(pos, i, dm):
    """Get angles"""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / dm)
    return pos * angle_rates


def positional_encoding(max_seq_len, dm):
    """calculates the positional encoding for a transformer

    -> max_seq_len is an integer representing the maximum sequence length
    -> dm is the model depth
    -> Returns: a numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encoding vectors
    """
    PE = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                    np.arange(dm)[np.newaxis, :], dm)

    PE[:, 0::2] = np.sin(PE[:, 0::2])

    PE[:, 1::2] = np.cos(PE[:, 1::2])

    return PE
