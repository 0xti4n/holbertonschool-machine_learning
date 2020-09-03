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
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    p_0 = padding[0]
    p_1 = padding[1]
    img = np.pad(images, ((0, 0), (p_0, p_0), (p_1, p_1)), mode='constant')
    output_h = image_h + 2 * p_0 - kh + 1
    output_w = image_w + 2 * p_1 - kw + 1
    output = np.zeros((m, output_h, output_w))
    for x in range(output_h):
        for y in range(output_w):
            slc = img[:, x: x + filter_h, y: y + filter_w]
            values = np.sum(slc * kernel, axis=1).sum(axis=1)
            output[:, x, y] = values
    return output
