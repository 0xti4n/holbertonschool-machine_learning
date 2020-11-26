#!/usr/bin/env python3
"""Vanilla Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates an autoencoder

    -> input_dims is an integer containing the
        dimensions of the model input

    -> hidden_layers is a list containing the number of
        nodes for each hidden layer in the encoder, respectively
        * the hidden layers should be reversed for the decoder

    -> latent_dims is an integer containing the dimensions
        of the latent space representation

    -> Returns: encoder, decoder, auto
        * encoder is the encoder model
        * decoder is the decoder model
        * auto is the full autoencoder model
    """
    # encoder
    x = keras.layers.Input(shape=(input_dims, ))
    x_prev = x
    for i in hidden_layers:
        encoded = keras.layers.Dense(i, activation='relu')(x_prev)
        x_prev = encoded

    latent = keras.layers.Dense(latent_dims,
                                activation='relu')(x_prev)

    # Decoder
    x_latent = keras.layers.Input(shape=(latent_dims, ))
    x_p = x_latent
    for l in reversed(hidden_layers):
        decoded = keras.layers.Dense(l, activation='relu')(x_p)
        x_p = decoded

    decoded = keras.layers.Dense(input_dims,
                                 activation='sigmoid')(x_p)

    encoder = keras.models.Model(x, latent)
    decoder = keras.models.Model(x_latent, decoded)

    out_decoder = decoder(encoder(x))
    autoencoder = keras.models.Model(x, out_decoder)
    autoencoder.compile(optimizer='adam',
                        loss='binary_crossentropy')

    return encoder, decoder, autoencoder
