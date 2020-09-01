#!/usr/bin/env python3
"""Input"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds a neural network with the Keras library"""
    inputs = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=reg)(inputs)
    for i in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=reg)(x)
    return K.Model(inputs=inputs, outputs=x)
