#!/usr/bin/env python3
""" Train"""
import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """builds, trains, and saves a neural network classifier"""
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    t_op = create_train_op(loss, alpha)

    tf.add_to_collections('x', x)
    tf.add_to_collections('y', y)
    tf.add_to_collections('y_pred', y_pred)
    tf.add_to_collections('accuracy', accuracy)
    tf.add_to_collections('loss', loss)
    tf.add_to_collections('train_op', t_op)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(iterations + 1):
            t_loss, t_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_train, y: Y_train})
            v_loss, v_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_valid, y: Y_valid})
            if i < iterations:
                sess.run(t_op, feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == 0 or i == iterations:
                print('After {} iterations:'.format(i))
                print('\tTraining Cost: {}'.format(t_loss))
                print('\tTraining Accuracy: {}'.format(t_accuracy))
                print('\tValidation Cost: {}'.format(v_loss))
                print('\tValidation Accuracy: {}'.format(v_accuracy))

        return saver.save(sess, save_path)
