#!/usr/bin/env python3
"""Convolution with Channels"""
import numpy as np
from math import ceil, floor


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a convolution on images with channels

    -> images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
        * c is the number of channels in the image
    -> kernel is a numpy.ndarray with shape (kh, kw) containing
    the kernel for the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel
        * nc is the number of kernels
    -> padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        -> if ‘same’, performs a same convolution
        -> if ‘valid’, performs a valid convolution
        -> if a tuple:
            -> ph is the padding for the height of the image
            -> pw is the padding for the width of the image
            -> the image should be padded with 0’s
    -> stride is a tuple of (sh, sw)
        -> sh is the stride for the height of the image
        -> sw is the stride for the width of the image
    """
    m = images.shape[0]
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    kh, kw = kernels.shape[0], kernels.shape[1]
    sh, sw = stride
    nc = kernels.shape[3]

    p_0 = 0
    p_1 = 0

    if padding == 'same':
        p_0 = int(((h - 1) * sh + kh - h) / 2) + 1
        p_1 = int(((w - 1) * sw + kw - w) / 2) + 1

    if type(padding) == tuple:
        p_0 = padding[0]
        p_1 = padding[1]

    output_h = int((h + 2 * p_0 - kh) / sh) + 1
    output_w = int((w + 2 * p_1 - kw) / sw) + 1
    output = np.zeros((m, output_h, output_w, nc))
    img = np.pad(images, ((0, 0), (p_0, p_0), (p_1, p_1), (0, 0)), 'constant')

    for ch in range(nc):
        for x in range(output_h):
            for y in range(output_w):
                slc = img[:, x * sh:sh * x + kh, y * sw: sw * y + kw, :]
                r = np.sum(kernels[..., ch] * slc,
                           axis=1).sum(axis=1).sum(axis=1)
                output[:, x, y, ch] = r
    return output
