#!/usr/bin/env python3
"""Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a convolutional
    layer of a neural network"""
    kh, kw, c_prev, c_new = W.shape
    _, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    sh, sw = stride

    p_0 = 0
    p_1 = 0

    if padding == 'same':
        p_0 = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        p_1 = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1

    output_h = int((h_prev + 2 * p_0 - kh) / sh) + 1
    output_w = int((w_prev + 2 * p_1 - kw) / sw) + 1
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    img = np.pad(A_prev, [(0, 0), (p_0, p_0), (p_1, p_1), (0, 0)], 'constant')

    dW = np.zeros(W.shape)
    dx = np.zeros(img.shape)

    for i in range(m):
        for h in range(output_h):
            for w in range(output_w):
                for k in range(c_new):
                    dx[i,
                        h * sh:h * sh + kh,
                        w * sw:w * sw + kw, :] += W[:,
                                                    :,
                                                    :,
                                                    k] * dZ[i, h, w, k]
                    dW[:,
                       :,
                       :,
                       k] += img[i,
                                 h * sw:h * sw + kh,
                                 w * sw:w * sw + kw, :] * dZ[i, h, w, k]

    if padding == 'same':
        dx = dx[:, p_0:-p_0, p_1:-p_1, :]

    return dx, dW, db
