#!/usr/bin/env python3
"""Convolution with Padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images
    with custom padding

    ->images is a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
    ->kernel is a numpy.ndarray with shape (kh, kw) containing
    the kernel for the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel
    ->padding is a tuple of (ph, pw)
        * ph is the padding for the height of the image
        * pw is the padding for the width of the image
    """
    m = images.shape[0]
    h, w = images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]

    p_0 = padding[0]
    p_1 = padding[1]

    img = np.pad(images, ((0, 0), (p_0, p_0), (p_1, p_1)), 'constant')

    output_h = h + 2 * P_0 - kh + 1
    output_w = w + 2 * P_1 - kw + 1
    output = np.zeros((m, output_h, output_w))

    for x in range(output_h):
        for y in range(output_w):
            slc = img[:, x:x + kh, y:y + kw]
            output[:, x, y] = np.sum(dlc * kernel, axis=1).sum(axis=1)
    return output
