#!/usr/bin/env python3
"""Strided Convolution """
import numpy as np
from math import floor


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on grayscale images
    ->images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        -> m is the number of images
        -> h is the height in pixels of the images
        -> w is the width in pixels of the images
    -> kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        -> kh is the height of the kernel
        -> kw is the width of the kernel
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
    h, w = images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride

    if padding == 'valid':
        p_0 = 0
        p_1 = 0
    elif padding == 'same':
        p_0 = int((kh - 1) / 2)
        p_1 = int((kw - 1) / 2)
    else:
        p_0 = padding[0]
        p_1 = padding[1]

    output_h = int(floor(float(h + 2 * p_0 - kh) / float(sh))) + 1
    output_w = int(floor(float(w + 2 * p_1 - kw) / float(sw))) + 1
    output = np.zeros((m, output_h, output_w))
    img = np.pad(images, ((0, 0), (p_0, p_0), (p_1, p_1)), 'constant')

    for x in range(output_h):
        for y in range(output_w):
            slc = img[:, x * sh:sh * x + kh, y * sw: sw * y + kw]
            output[:, x, y] = np.sum(kernel * slc, axis=1).sum(axis=1)
    return output
