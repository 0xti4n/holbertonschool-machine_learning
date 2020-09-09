#!/usr/bin/env python3
"""Pooling Forward Prop """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """"""

    m = A_prev.shape[0]
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = int((h_prev - kh) / sh) + 1
    output_w = int((w_prev - kw) / sw) + 1
    output = np.zeros((m, output_h, output_w, c_prev))

    for x in range(output_h):
        for y in range(output_w):
            slc = A_prev[:, x * sh:sh * x + kh, y * sw: sw * y + kw]
            if mode == 'max':
                output[:, x, y] = slc.max(axis=1).max(axis=1)
            elif mode == 'avg':
                output[:, x, y] = slc.mean(axis=1).mean(axis=1)
    return output
