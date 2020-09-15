#!/usr/bin/env python3
"""Dense Block"""
import tensorflow.keras as K


def conv_block(X, growth_rate):
    """normalization, activation and
    convolution proccess"""
    x = K.layers.BatchNormalization(axis=3)(X)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(filters=4 * growth_rate,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer='he_normal')(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(filters=growth_rate,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer='he_normal')(x)
    return x


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in Densely Connected
    Convolutional Networks:

    -> X is the output from the previous layer
    -> nb_filters is an integer representing the number of filters in X
    -> growth_rate is the growth rate for the dense block
    -> layers is the number of layers in the dense block
    -> You should use the bottleneck layers used for DenseNet-B
    -> All weights should use he normal initialization
    -> All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    -> Returns: The concatenated output of each layer within the Dense
    Block and the number of filters within the concatenated outputs,
    respectively

    """
    concat_input = X
    for _ in range(layers):
        x = conv_block(concat_input, growth_rate)
        concat_input = K.layers.concatenate([concat_input, x], axis=3)
        nb_filters += growth_rate
    return concat_input, nb_filters
