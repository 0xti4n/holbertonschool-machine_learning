#!/usr/bin/env python3
"""Pooling Backward Prop """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='avg'):
    """performs back propagation over a pooling
    layer of a neural network

    -> dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the
    output of the pooling layer
        * m is the number of examples
        * h_new is the height of the output
        * w_new is the width of the output
        * c is the number of channels

    -> A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
    containing the output of the previous layer
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer

    -> kernel_shape is a tuple of (kh, kw) containing the size of the
    kernel for the pooling
        * kh is the kernel height
        * kw is the kernel width

    -> stride is a tuple of (sh, sw) containing the strides for the pooling
        * sh is the stride for the height
        * sw is the stride for the width

    -> mode is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively

    -> Returns: the partial derivatives with respect to the
    previous layer (dA_prev)
    """
    kh, kw = kernel_shape
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    sh, sw = stride

    output_h = int((h_prev - kh) / sh) + 1
    output_w = int((w_prev - kw) / sw) + 1

    pd = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(output_h):
            for w in range(output_w):
                for k in range(c):
                    if mode == 'max':
                        tmp = A_prev[i, h * sh:h * sh + kh,
                                     w * sw:w * sw + kh, k]
                        mask = tmp == tmp.max()
                        pd[i, h * sh:h * sh + kh,
                           w * sw:w * sw + kw, k] += dA[i, h, w, k] * mask
                    if mode == 'avg':
                        pd[i, h * sh:h * sh + kh,
                           w * sw:w * sw + kw, k] += dA[i, h, w, k] / (kh * kw)
    return pd
