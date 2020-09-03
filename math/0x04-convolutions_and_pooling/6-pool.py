#!/usr/bin/env python3
"""Strided Convolution """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs a convolution on grayscale images
    -> images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
        -> m is the number of images
        -> h is the height in pixels of the images
        -> w is the width in pixels of the images
        -> c is the number of channels in the image
    -> kernel_shape is a tuple of (kh, kw) containing the
    kernel shape for the pooling
        -> kh is the height of the kernel
        -> kw is the width of the kernel
    -> stride is a tuple of (sh, sw)
        -> sh is the stride for the height of the image
        -> sw is the stride for the width of the image
    -> mode indicates the type of pooling
        -> max indicates max pooling
        -> avg indicates average pooling
    """
    m = images.shape[0]
    h, w = images.shape[1], images.shape[2]
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride
    channels = images.shape[3]

    output_h = int(floor(float(h - kh) / float(sh))) + 1
    output_w = int(floor(float(w - kw) / float(sw))) + 1
    output = np.zeros((m, output_h, output_w, channels))

    for x in range(output_h):
        for y in range(output_w):
            slc = images[:, x * sh:sh * x + kh, y * sw: sw * y + kw, :]
            if mode == 'max':
                output[:, x, y] = slc.max(axis=(1, 2))
            elif mode == 'avg':
                output[:, x, y, :] = slc.mean(axis=(1, 2))
    return output