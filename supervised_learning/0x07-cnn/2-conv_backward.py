#!/usr/bin/env python3
"""Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a convolutional
    layer of a neural network"""
    hk = W.shape[0]
    wk = W.shape[1]
    nc = W.shape[3]
    m = A_prev.shape[0]
    hm = A_prev.shape[1]
    wm = A_prev.shape[2]
    st1 = stride[1]
    st0 = stride[0]

    p_0 = 0
    p_1 = 0

    if padding == 'same':
        p_0 = int(((hm - 1) * st0 + hk - hm) / 2) + 1
        p_1 = int(((wm - 1) * st1 + wk - wm) / 2) + 1
    
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    out_h = int((hm + 2 * p_0 - hk) / st0) + 1
    out_w = int((wm + 2 * p_1 - wk) / st1) + 1
    img = np.pad(A_prev, ((0, 0), (p_0, p_0), (p_1, p_1), (0, 0)), 'constant')
    
    dX = np.zeros(img.shape)
    dW = np.zeros(W.shape)

    for i in range(m):
        for h in range(out_h):
            for w in range(out_w):
                for c in range(nc):
                    dX[i, h * st0: h * st0 + hk,
                       w * st1: w * st1 + wk, :] += dZ[
                       i, h, w, c] * W[:, :, :, c]
                    dW[:, :, :, c] += img[i, h * st0: h * st0 + hk,
                                          w * st1: w * st1 + wk,
                                          :] * dZ[i, h, w, c]
    if padding == "same":
        dX = dX[:, pad0:-pad0, pad1:-pad1, :]

    return dX, dW, db
