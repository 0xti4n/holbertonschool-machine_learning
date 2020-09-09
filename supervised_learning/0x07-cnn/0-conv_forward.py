#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional
    layer of a neural network

    -> A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        * m is the number of examples
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
        * c_prev is the number of channels in the previous layer

    -> W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
    containing the kernels for the convolution
        * kh is the filter height
        * kw is the filter width
        * c_prev is the number of channels in the previous layer
        * c_new is the number of channels in the output

    -> b is a numpy.ndarray of shape (1, 1, 1, c_new)
    containing the biases applied to the convolution

    -> activation is an activation function applied to the convolution

    -> padding is a string that is either same or valid, indicating
    the type of padding used

    -> stride is a tuple of (sh, sw) containing the strides for the convolution
        * sh is the stride for the height
        * sw is the stride for the width

    -> Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    p_0 = 0
    p_1 = 0

    if padding == 'same':
        p_0 = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        p_1 = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1

    output_h = int((h_prev + 2 * p_0 - kh) / sh) + 1
    output_w = int((w_prev + 2 * p_1 - kw) / sw) + 1
    output = np.zeros((m, output_h, output_w, c_new))
    img = np.pad(A_prev, ((0, 0), (p_0, p_0), (p_1, p_1), (0, 0)), 'constant')

    for ch in range(c_new):
        for x in range(output_h):
            for y in range(output_w):
                slc = img[:, x * sh:sh * x + kh, y * sw: sw * y + kw, :]
                r = np.sum(W[..., ch] * slc,
                           axis=1).sum(axis=1).sum(axis=1)
                output[:, x, y, ch] = activation(r + b[0, 0, 0, ch])

    return output
