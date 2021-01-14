#!/usr/bin/env python3
"""Full transformer network train"""
import tensorflow.compat.v2 as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """CUstom LR"""
    def __init__(self, dm, warmup_steps=4000):
        """class init"""
    super(LRSchedule, self).__init__()
    self.dm = dm
    self.dm = tf.cast(self.dm, tf.float32)

    self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Train transformer network"""
    data = Dataset(batch_size, max_len)

    in_vocab = data.tokenizer_pt.vocab_size + 2
    tar_vocab = data.tokenizer_en.vocab_size + 2
    rate = 0.1

    transformer = Transformer(N, dm, h,
                              hidden,
                              in_vocab,
                              tar_vocab,
                              in_vocab,
                              tar_vocab,
                              rate)

    lr = LRSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(lr,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy
    loss_object = loss_object(from_logits=True, reduction='none')

    def loss_function(real, pred):
        """Loss function"""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def acc_function(real, pred):
        """accuracy function"""
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='loss')
    train_acc = tf.keras.metrics.Mean(name='train_accuracy')

    for epoch in range(epochs):
        for (batch, (inp, tar)) in enumerate(data.data_train):
            tar_input = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_mask, look_mask, dec_mask = create_masks(inp, tar_input)

            with tf.GradientTape() as tape:
                predictions = transformer(inp,
                                          tar_input,
                                          True,
                                          enc_mask,
                                          look_mask,
                                          dec_mask)

                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients,
                                          transformer.trainable_variables))

            train_loss(loss)
            train_acc(acc_function(tar_real, predictions))

            if batch % 50 == 0:
                print('Epoch {} \
                      batch {} \
                      loss {} \
                      accuracy {}'.format(epoch + 1, batch,
                                          train_loss.result(),
                                          train_acc.result()))

    print('Epoch {} loss {} accuracy {}'.format(epoch + 1,
                                                train_loss.result(),
                                                train_acc.result()))
    return transformer
