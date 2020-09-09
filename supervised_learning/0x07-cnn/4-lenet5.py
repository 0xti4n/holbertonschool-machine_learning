#!/usr/bin/env python3
"""LeNet-5 (Tensorflow) """
import tensorflow as tf


def lenet5(x, y):
    """builds a modified version of the
    LeNet-5 architecture using tensorflow

    -> x is a tf.placeholder of shape (m, 28, 28, 1) containing
    the input images for the network
        * m is the number of images

    -> y is a tf.placeholder of shape (m, 10) containing the one-hot
    labels for the network

    -> The model should consist of the following layers in order:
        * Convolutional layer with 6 kernels of shape 5x5 with same padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Convolutional layer with 16 kernels of shape 5x5 with valid padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Fully connected layer with 120 nodes
        * Fully connected layer with 84 nodes
        * Fully connected softmax output layer with 10 nodes

    -> All layers requiring initialization should initialize
    their kernels with the he_normal initialization
    method: tf.contrib.layers.variance_scaling_initializer()

    -> All hidden layers requiring activation should use the relu
    activation function
    -> you may import tensorflow as tf
    -> you may NOT use tf.keras
    -> Returns:
        * a tensor for the softmax activated output
        * a training operation that utilizes Adam optimization
            (with default hyperparameters)
        * a tensor for the loss of the netowrk
        * a tensor for the accuracy of the network
    """

    init = tf.contrib.layers.variance_scaling_initializer()

    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=init,
                             activation=tf.nn.relu)(x)

    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=5,
                             padding='valid',
                             kernel_initializer=init,
                             activation=tf.nn.relu)(pool1)

    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv2)

    flat = tf.layers.Flatten()(pool2)

    dense1 = tf.layers.Dense(units=120,
                             activation=tf.nn.relu,
                             kernel_initializer=init,)(flat)

    dense2 = tf.layers.Dense(units=84,
                             activation=tf.nn.relu,
                             kernel_initializer=init,)(dense1)

    out = tf.layers.Dense(units=10,
                          kernel_initializer=init)(dense2)

    out_act = tf.nn.softmax(out)

    cost = tf.losses.softmax_cross_entropy(y, out)
    correct_prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(cost)

    return out_act, train_op, cost, accuracy
