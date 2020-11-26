#!/usr/bin/env python3
"""Convolutional Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder

    -> input_dims is a tuple of integers containing
        the dimensions of the model input

    -> filters is a list containing the number of
        filters for each convolutional layer in the encoder, respectively

        * the filters should be reversed for the decoder

    -> latent_dims is a tuple of integers containing
        the dimensions of the latent space representation

    -> Each convolution in the encoder should use a kernel
        size of (3, 3) with same padding and relu activation,
            followed by max pooling of size (2, 2)

    -> Each convolution in the decoder, except for the last
        two, should use a filter size of (3, 3) with same
        padding and relu activation, followed by upsampling of size (2, 2)

        * The second to last convolution should instead use valid padding
        * The last convolution should have the same number of
        filters as the number of channels in input_dims with
        sigmoid activation and no upsampling

    -> Returns: encoder, decoder, auto
        * encoder is the encoder model
        * decoder is the decoder model
        * auto is the full autoencoder model
    """
    # encoder
    input_img = keras.layers.Input(shape=input_dims)
    encoded = input_img
    for i in filters:
        encoded = keras.layers.Conv2D(i,
                                      (3, 3),
                                      activation='relu',
                                      padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)

    # decoder
    x_latent = keras.layers.Input(shape=latent_dims)
    decoded = x_latent
    for i, f in enumerate(reversed(filters)):
        if i == len(filters) - 1:
            decoded = keras.layers.Conv2D(f, (3, 3),
                                          activation='relu',
                                          padding='valid')(decoded)
        else:
            decoded = keras.layers.Conv2D(f, (3, 3),
                                          activation='relu',
                                          padding='same')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                  activation='sigmoid',
                                  padding='same')(decoded)

    encoder = keras.models.Model(input_img, encoded)
    decoder = keras.models.Model(x_latent, decoded)
    out_decoder = decoder(encoder(input_img))
    autoencoder = keras.models.Model(input_img, out_decoder)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
