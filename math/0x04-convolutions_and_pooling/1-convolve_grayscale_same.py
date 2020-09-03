#!/usr/bin/env python3
"""Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a valid convolution on
    grayscale images

    -> images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
    -> kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel

    -> Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h, w = images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]

    p_0 = int(kh / 2)
    p_1 = int(kw / 2)
    output_h = h
    output_w = w

    output = np.zeros((m, output_h, output_w))
    img = np.pad(images, ((0, 0), (p_0, p_0), (p_1, p_1)), 'constant')

    for x in range(output_h):
        for y in range(output_w):
            slc = img[:, x:x + kh, y:y + kw]
            output[:, x, y] = np.sum(kernel * slc, axis=1).sum(axis=1)
    return output
