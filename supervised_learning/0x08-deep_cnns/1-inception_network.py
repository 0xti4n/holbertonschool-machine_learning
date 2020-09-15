#!/usr/bin/env python3
"""Inception Network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """builds the inception network as
    described in Going Deeper with
    Convolutions (2014)

    Returns: the keras model
    """

    X = K.Input(shape=(224, 224, 3))

    block_1 = K.layers.Conv2D(filters=64,
                              kernel_size=7,
                              activation='relu',
                              padding='same',
                              strides=(2, 2))(X)

    block_2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    padding='same',
                                    strides=(2, 2))(block_1)

    block_3 = K.layers.Conv2D(filters=64,
                              kernel_size=1,
                              activation='relu',
                              padding='same',
                              strides=(1, 1))(block_2)

    block_4 = K.layers.Conv2D(filters=192,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              strides=(1, 1))(block_3)

    block_5 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    padding='same',
                                    strides=(2, 2))(block_4)

    block_6 = inception_block(block_5, [64, 96, 128, 16, 32, 32])

    block_7 = inception_block(block_6, [128, 128, 192, 32, 96, 64])

    block_8 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    padding='same',
                                    strides=(2, 2))(block_7)

    block_9 = inception_block(block_8, [192, 96, 208, 16, 48, 64])

    block_10 = inception_block(block_9, [160, 112, 224, 24, 64, 64])

    block_11 = inception_block(block_10, [128, 128, 256, 24, 64, 64])

    block_12 = inception_block(block_11, [112, 144, 288, 32, 64, 64])

    block_13 = inception_block(block_12, [256, 160, 320, 32, 128, 128])

    block_14 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     padding='same',
                                     strides=(2, 2))(block_13)

    block_15 = inception_block(block_14, [256, 160, 320, 32, 128, 128])

    block_16 = inception_block(block_15, [384, 192, 384, 48, 128, 128])

    block_17 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1))(block_16)

    block_18 = K.layers.Dropout(rate=0.4)(block_17)

    block_19 = K.layers.Dense(units=1000, activation='softmax')(block_18)

    model = K.Model(X, block_19)

    return model
