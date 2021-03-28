#!/usr/bin/env python3
"""PCA image alexnet"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """performs PCA color augmentation as described in the AlexNet paper

    Args:
    -> image is a 3D tf.Tensor containing the image to change
    -> alphas a tuple of length 3 containing the amount that
    each channel should change

    Returns:
    the augmented image
    """
    img = tf.keras.preprocessing.image.img_to_array(image)
    img_orig = img.astype(float).copy()

    img = img / 255.0
    img_rs = img.reshape(-1, 3)

    img_center = img_rs - np.mean(img_rs, axis=0)

    img_cov = np.cov(img_center, rowvar=False)

    e_val, e_vects = np.linalg.eigh(img_cov)

    sort_perm = e_val[::-1].argsort()
    e_val[::-1].sort()
    e_vects = e_vects[:, sort_perm]

    m1 = np.column_stack((e_vects))
    m2 = np.zeros((3, 1))
    m2[:, 0] = alphas * e_val[:]

    add_vect = np.matrix(m1) * np.matrix(m2)

    for i in range(3):
        img_orig[..., i] += add_vect[i]

    img_orig = np.clip(img_orig, 0.0, 255.0)
    img_orig = img_orig.astype(np.uint8)

    return img_orig
