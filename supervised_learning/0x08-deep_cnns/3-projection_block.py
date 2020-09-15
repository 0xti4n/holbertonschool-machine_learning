#!/usr/bin/env python3
"""Projection Block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015):


    -> A_prev is the output from the previous layer
    -> filters is a tuple or list containing F11, F3, F12, respectively:
            *F11 is the number of filters in the first 1x1 convolution
            *F3 is the number of filters in the 3x3 convolution
            *F12 is the number of filters in the second 1x1 convolution as
            well as the 1x1 convolution in the shortcut connection
    -> s is the stride of the first convolution in both the main path and
    the shortcut connection

    -> All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.

    -> All weights should use he normal initialization

    -> Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters
    X_SHORTCOUT = A_prev
    X = K.layers.Conv2D(filters=F11,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        strides=(s, s))(A_prev)

    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Activation(activation='relu')(X)

    X = K.layers.Conv2D(filters=F3,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Activation(activation='relu')(X)

    X = K.layers.Conv2D(filters=F12,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer='he_normal')(X)

    X = K.layers.BatchNormalization(axis=3)(X)

    X_SHORTCOUT = K.layers.Conv2D(filters=F12,
                                  kernel_size=1,
                                  padding='same',
                                  kernel_initializer='he_normal',
                                  strides=(s, s))(X_SHORTCOUT)

    X_SHORTCOUT = K.layers.BatchNormalization(axis=3)(X_SHORTCOUT)

    X = K.layers.Add()([X, X_SHORTCOUT])
    X = K.layers.Activation(activation='relu')(X)

    return X
