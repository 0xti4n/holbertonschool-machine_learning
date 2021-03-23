#!/usr/bin/env python3
"""Shear image randomly"""
import tensorflow as tf


def shear_image(image, intensity):
    """randomly shears an image:

    Args:
    -> image is a 3D tf.Tensor containing the image to shear
    -> intensity is the intensity with which the image should
    be sheared

    Returns:
    -> the sheared image
    """
    pre = tf.keras.preprocessing
    pre_img = pre.image.img_to_array
    pre_arr_to_img = pre.image.array_to_img
    img = pre_img(image)
    shear = pre.image.random_shear(
        x=img,
        intensity=intensity,
        row_axis=1,
        col_axis=0,
        channel_axis=2
    )

    return pre_arr_to_img(shear)
