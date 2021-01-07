#!/usr/bin/env python3
"""Self Attention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Class that calculate the attention
        for machine translation"""
    def __init__(self, units):
        """
        -> units is an integer representing the number of hidden
            units in the alignment model
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        -> s_prev is a tensor of shape (batch, units)
            containing the previous decoder hidden state
        -> hidden_states is a tensor of shape
            batch, input_seq_len, units)containing the outputs
                of the encoder

        -> Returns: context, weights
            * context is a tensor of shape (batch, units) that contains
                the context vector for the decoder
            * weights is a tensor of shape (batch, input_seq_len, 1)
                that contains the attention weights
        """
        s_prev = tf.expand_dims(s_prev, 1)
        e = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))

        weights = tf.nn.softmax(e, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
