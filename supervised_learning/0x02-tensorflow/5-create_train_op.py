#!/usr/bin/env python3
""" Train_op"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """creates the training operation
    for the network

    - loss is the loss of the network’s prediction
    - alpha is the learning rate
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
