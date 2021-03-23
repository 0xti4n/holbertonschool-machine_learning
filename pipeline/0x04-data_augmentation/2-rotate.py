#!/usr/bin/env python3
"""rotate image"""
import tensorflow as tf


def rotate_image(image):
    """rotates an image by 90 degrees counter-clockwise:

    Args:
    -> image is a 3D tf.Tensor containing the image to rotate

    Returns
    -> the rotated image
    """
    rot = tf.image.rot90(image)

    return rot
