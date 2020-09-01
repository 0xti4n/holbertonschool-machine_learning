#!/usr/bin/env python3
"""Sequential"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i],
                      activation=activations[i],
                      kernel_regularizer=reg,
                      input_dim=nx,
                      name='dense'))
        else:
            model.add(K.layers.Dense(layers[i],
                      activation=activations[i],
                      kernel_regularizer=reg,
                      name='dense_{}'.format(str(i))))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(keep_prob))

    return model
