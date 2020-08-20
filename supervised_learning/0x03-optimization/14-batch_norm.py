#!/usr/bin/env python3
""" Batch Normalization"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer
    for a neural network in tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=None,
                            name='layer',
                            kernel_initializer=init)

    out = layer(prev)
    mean, var = tf.nn.moments(out, axes=0, keep_dims=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    norm = tf.nn.batch_normalization(out, mean, var, offset=beta, scale=gamma,
                                     variance_epsilon=1e-8)

    return activation(norm)
