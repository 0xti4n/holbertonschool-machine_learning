#!/usr/bin/env python3
"""Train"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """train the model using early stopping"""
    params = []
    if validation_data and early_stopping:
        early = K.callbacks.EarlyStopping(monitor='val_loss',
                                          mode='min',
                                          patience=patience)
        params.append(early)

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=[params])
    return history
