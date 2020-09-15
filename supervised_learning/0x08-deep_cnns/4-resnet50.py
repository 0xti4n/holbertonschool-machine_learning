#!/usr/bin/env python3
"""ResNet-50"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """builds the ResNet-50 architecture
    as described in Deep Residual Learning
    for Image Recognition (2015)

    Returns: the keras model
    """
    X_1 = K.Input(shape=(224, 224, 3))
    X = K.layers.Conv2D(filters=64,
                        kernel_size=7,
                        padding='same',
                        kernel_initializer='he_normal',
                        strides=(2, 2))(X_1)

    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Activation(activation='relu')(X)

    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              padding='same',
                              strides=(2, 2))(X)

    X = projection_block(X, [64, 64, 256], 1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])
    X = projection_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = projection_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = projection_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(1, 1))(X)

    X = K.layers.Dense(units=1000, activation='softmax')(X)

    model = K.Model(X_1, X)

    return model
