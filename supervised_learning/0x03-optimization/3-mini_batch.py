#!/usr/bin/env python3
"""Mini-Batch"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """trains a loaded neural network model using
    mini-batch gradient descent"""
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(load_path + '.meta')
        new_saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        iterations = X_train.shape[0] // batch_size
        if iterations % batch_size != 0:
            iterations += 1
            flag = True
        else:
            flag = False

        for i in range(epochs + 1):
            cost_t, acc_t = sess.run([loss, accuracy],
                                     {x: X_train, y: Y_train})
            cost_v, acc_v = sess.run([loss, accuracy],
                                     {x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(cost_t))
            print('\tTraining Accuracy: {}'.format(acc_t))
            print('\tValidation Cost: {}'.format(cost_v))
            print('\tValidation Accuracy: {}'.format(acc_v))

            if i < epochs:
                x_t, y_t = shuffle_data(X_train, Y_train)

                for j in range(iterations):
                    start = j * batch_size
                    if j == iterations - 1 and flag:
                        end = X_train.shape[0]
                    else:
                        end = j * batch_size + batch_size
                    b_x = x_t[start:end]
                    b_y = y_t[start:end]
                    sess.run(train_op, {x: b_x, y: b_y})
                    if j != 0 and (j + 1) % 100 == 0:
                        new_c, new_ac = sess.run([loss, accuracy],
                                                 {x: b_x, y: b_y})
                        print('\tStep {}:'.format(j + 1))
                        print('\t\tCost: {}'.format(new_c))
                        print('\t\tAccuracy: {}'.format(new_ac))
        return new_saver.save(sess, save_path)
