#!/usr/bin/env python3
"""Inception Block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """inception block as described
    in Going Deeper with Convolutions (2014):

    -> A_prev is the output from the previous layer
    -> filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
        respectively:
        *F1 is the number of filters in the 1x1 convolution
        *F3R is the number of filters in the 1x1 convolution
        before the 3x3 convolution
        *F3 is the number of filters in the 3x3 convolution
        *F5R is the number of filters in the 1x1 convolution
        before the 5x5 convolution
        *F5 is the number of filters in the 5x5 convolution
        *FPP is the number of filters in the 1x1 convolution
        after the max pooling

    -> All convolutions inside the inception block should use a
    rectified linear activation (ReLU)

    -> Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    block_1 = K.layers.Conv2D(filters=F1,
                              kernel_size=1,
                              activation='relu',
                              padding='same')(A_prev)

    block_2 = K.layers.Conv2D(filters=F3R,
                              kernel_size=1,
                              activation='relu',
                              padding='same')(A_prev)

    block_2 = K.layers.Conv2D(filters=F3,
                              kernel_size=3,
                              activation='relu',
                              padding='same')(block_2)

    block_3 = K.layers.Conv2D(filters=F5R,
                              kernel_size=1,
                              activation='relu',
                              padding='same')(A_prev)

    block_3 = K.layers.Conv2D(filters=F5,
                              kernel_size=5,
                              activation='relu',
                              padding='same')(block_3)

    block_4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    padding='same',
                                    strides=(1, 1))(A_prev)

    block_4 = K.layers.Conv2D(filters=FPP,
                              kernel_size=1,
                              activation='relu',
                              padding='same')(block_4)

    output = K.layers.concatenate([block_1, block_2, block_3, block_4],
                                  axis=3)

    return output
