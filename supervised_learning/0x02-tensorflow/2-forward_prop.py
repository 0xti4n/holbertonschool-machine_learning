#!/usr/bin/env python3
"""Forward Propagation """
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward Propagation function"""
    x_input = x
    for i in range(len(layer_sizes)):
        x_input = create_layer(x_input, layer_sizes[i], activations[i])
    return x_input
