#!/usr/bin/env python3
"""Initialize Q-table"""
import numpy as np


def q_init(env):
    """ initializes the Q-table

    Args:
    -> env is the FrozenLakeEnv instance

    Returns: the Q-table as a numpy.ndarray of zeros
    """
    x, y = env.desc.shape
    z = env.P[0]
    q_table = np.zeros((x*y, len(z)))

    return q_table
