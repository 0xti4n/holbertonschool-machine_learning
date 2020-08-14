#!/usr/bin/env python3
""" Layers"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Function that creates layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            name='layer',
                            kernel_initializer=init)
    output = layer(prev)
    return output
