#!/usr/bin/env python3
"""Full transformer network"""
import tensorflow.compat.v2 as tf
import numpy as np


def get_angles(pos, i, dm):
    """Get angles"""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / dm)
    return pos * angle_rates


def positional_encoding(max_seq_len, dm):
    """calculates the positional encoding for a transformer

    -> max_seq_len is an integer representing the maximum sequence length
    -> dm is the model depth
    -> Returns: a numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encoding vectors
    """
    PE = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                    np.arange(dm)[np.newaxis, :], dm)

    PE[:, 0::2] = np.sin(PE[:, 0::2])

    PE[:, 1::2] = np.cos(PE[:, 1::2])

    return PE


def sdp_attention(Q, K, V, mask=None):
    """calculates the scaled dot product attention:

    -> Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
        containing the query matrix
    -> K is a tensor with its last two dimensions as (..., seq_len_v, dk)
        containing the key matrix
    -> V is a tensor with its last two dimensions as (..., seq_len_v, dv)
        containing the value matrix
    -> mask is a tensor that can be broadcast into (..., seq_len_q, seq_len_v)
        containing the optional mask, or defaulted to None

    -> The preceding dimensions of Q, K, and V are the same

    -> Returns: output, weights
        * output a tensor with its last two dimensions as (..., seq_len_q, dv)
            containing the scaled dot product attention
        * weights a tensor with its last two dimensions as
            (..., seq_len_q, seq_len_v) containing the attention weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Class perform multi head attention"""

    def __init__(self, dm, h):
        """
        -> dm is an integer representing the dimensionality of the model
        -> h is an integer representing the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // self.h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        -> Split the last dimension into (num_heads, depth).
        -> Transpose the result such that the shape is
            (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        -> Q is a tensor of shape (batch, seq_len_q, dk) containing
            the input to generate the query matrix
        -> K is a tensor of shape (batch, seq_len_v, dk) containing
            the input to generate the key matrix
        -> V is a tensor of shape (batch, seq_len_v, dv) containing
            the input to generate the value matrix
        -> mask is always None

        -> Returns: output, weights
            * output a tensor with its last two dimensions as
                (..., seq_len_q, dm) containing the scaled
                dot product attention
            * weights a tensor with its last three dimensions as
                (..., h, seq_len_q, seq_len_v) containing the
                attention weights
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_att, weights = sdp_attention(Q, K, V, mask)

        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])
        concat_att = tf.reshape(scaled_att, (batch_size, -1,  self.dm))

        output = self.linear(concat_att)

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """Class Encoder"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        -> dm - the dimensionality of the model
        -> h - the number of heads
        -> hidden - the number of hidden units in the fully connected layer
        -> drop_rate - the dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        -> x - a tensor of shape (batch, input_seq_len, dm)
            containing the input to the encoder block
        -> training - a boolean to determine if the model is training
        -> mask - the mask to be applied for multi head attention

        -> Returns: a tensor of shape (batch, input_seq_len, dm)
            containing the block’s output
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(x + attn_output)
        forward_output = self.dense_hidden(out1)
        forward_output = self.dense_output(forward_output)

        forward_output = self.dropout2(forward_output, training=training)

        output = self.layernorm2(out1 + forward_output)

        return output


class DecoderBlock(tf.keras.layers.Layer):
    """Class to create an encoder block
        for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        -> dm - the dimensionality of the model
        -> h - the number of heads
        -> hidden - the number of hidden units in the fully connected layer
        -> drop_rate - the dropout rate
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        -> x - a tensor of shape (batch, target_seq_len, dm)
            containing the input to the decoder block
        -> encoder_output - a tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder
        -> training - a boolean to determine if the model is training
        -> look_ahead_mask - the mask to be applied to the first multi
            head attention layer
        -> padding_mask - the mask to be applied to the second multi
            head attention layer

        -> Returns: a tensor of shape (batch, target_seq_len, dm)
            containing the block’s output
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(out1,
                             encoder_output,
                             encoder_output,
                             padding_mask)

        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        forward_output = self.dense_hidden(out2)
        forward_output = self.dense_output(forward_output)
        forward_output = self.dropout3(forward_output, training=training)

        output = self.layernorm3(out2 + forward_output)

        return output


class Encoder(tf.keras.layers.Layer):
    """Class to create the encoder for a transformer"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        -> N - the number of blocks in the encoder
        -> dm - the dimensionality of the model
        -> h - the number of heads
        -> hidden - the number of hidden units in the fully connected layer
        -> input_vocab - the size of the input vocabulary
        -> max_seq_len - the maximum sequence length possible
        -> drop_rate - the dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=self.dm)

        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        -> x - a tensor of shape (batch, input_seq_len, dm)
            containing the input to the encoder

        -> training - a boolean to determine if the model is training
        -> mask - the mask to be applied for multi head attention

        -> Returns: a tensor of shape (batch, input_seq_len, dm)
            containing the encoder output
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """Class to create the decoder for a transformer"""

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        -> N - the number of blocks in the encoder
        -> dm - the dimensionality of the model
        -> h - the number of heads
        -> hidden - the number of hidden units in the fully connected layer
        -> target_vocab - the size of the target vocabulary
        -> max_seq_len - the maximum sequence length possible
        -> drop_rate - the dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=self.dm)

        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """
        -> x - a tensor of shape (batch, target_seq_len, dm)
            containing the input to the decoder
        -> encoder_output - a tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder
        -> training - a boolean to determine if the model is training
        -> look_ahead_mask - the mask to be applied to the first multi
            head attention layer
        -> padding_mask - the mask to be applied to the second multi
            head attention layer

        -> Returns: a tensor of shape (batch, target_seq_len, dm)
            containing the decoder output
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output,
                               training,
                               look_ahead_mask,
                               padding_mask)

        return x


class Transformer(tf.keras.layers.Layer):
    """Class to create a transformer network"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """
        -> N - the number of blocks in the encoder and decoder
        -> dm - the dimensionality of the model
        -> h - the number of heads
        -> hidden - the number of hidden units in the fully connected layers
        -> input_vocab - the size of the input vocabulary
        -> target_vocab - the size of the target vocabulary
        -> max_seq_input - the maximum sequence length possible
            for the input
        -> max_seq_target - the maximum sequence length possible
                for the target
        -> drop_rate - the dropout rate
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)

        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)

        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        -> inputs - a tensor of shape (batch, input_seq_len)
            containing the inputs
        -> target - a tensor of shape (batch, target_seq_len)
            containing the target
        -> training - a boolean to determine if the model is training
        -> encoder_mask - the padding mask to be applied to the encoder
        -> look_ahead_mask - the look ahead mask to be applied to the decoder
        -> decoder_mask - the padding mask to be applied to the decoder

        -> Returns: a tensor of shape (batch, target_seq_len, target_vocab)
            containing the transformer output
        """
        enc_out = self.encoder(inputs, training, encoder_mask)
        dec_out = self.decoder(target, enc_out, training,
                               look_ahead_mask, decoder_mask)

        output = self.linear(dec_out)

        return output
