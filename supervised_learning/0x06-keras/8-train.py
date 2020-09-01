#!/usr/bin/env python3
"""Train"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """save the best iteration of the model"""
    def lr_scheduler(epoch):
        return alpha / (1 + decay_rate * epoch)

    params = []

    if validation_data and early_stopping:
        params.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                mode='min',
                                                patience=patience))

    if validation_data and learning_rate_decay:
        params.append(K.callbacks.LearningRateScheduler(lr_scheduler,
                                                        verbose=1))

    if save_best:
        params.append(K.callbacks.ModelCheckpoint(filepath,
                                                  save_best_only=True))

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=params)

    return history
