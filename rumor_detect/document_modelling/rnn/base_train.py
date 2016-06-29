# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import tensorflow as tf


def train(loss, global_step, decay_steps,
          initial_learning_rate=0.01, decay_rate=0.1):
    # create learning rate
    lr = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    tf.scalar_summary('learning rate', lr)

    # compute and apply gradients
    opt = tf.train.MomentumOptimizer(lr, 0.9)
    grads = opt.compute_gradients(loss)
    apply_grads = opt.apply_gradients(grads, global_step)

    # add histograms for variables and gradients
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradient', grad)

    # smoothing loss and variables
    ema = tf.train.ExponentialMovingAverage(0.999, global_step)
    ema_op = ema.apply(tf.trainable_variables() + [loss])
    tf.scalar_summary('moving_loss', ema.average(loss))

    # combine all op into a train_op
    with tf.control_dependencies([apply_grads]):
        train_op = tf.group(ema_op, name='train')

    return train_op
