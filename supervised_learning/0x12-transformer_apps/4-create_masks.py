#!/usr/bin/env python3
"""Create masks for transformer"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """creates all masks for training/validation"""

    size = tf.shape(target)[1]
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_mask, look_ahead_mask, encoder_mask
