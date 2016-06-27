# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import tensorflow as tf
from parameter import *

def activation_summary_(x):
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def moving_loss_(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.999, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + '(raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def variable_init_gaussian_with_weight_decay_(name, shape, stddev, wd=None):
    """
    Helper to create an initialized variable with weight decay.

    Input
    - name: name of the variable.
    - shape: shape of the variable.
    - stddev: standard deviation of a truncated Gaussian.
    - wd: weight decay for the variable. If wd=None, no weight decay is applied.
    Output
    - var: tensor variable.
    """
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def variable_init_constant_(name, shape, val=.0):
    """
    Helper to create an variable. Initalized all to val.
    Input
    - name: name of the variable.
    - shape: shape of the variable.
    - val: value to initialize.
    Output
    - var: tensor variable.
    """
    initializer = tf.constant_initializer(val)
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def input_placeholder():
    X = tf.placeholder(tf.int32, shape=[None, SEQUENCE_LENGTH], name="input_X")
    y = tf.placeholder(tf.int32, shape=[None], name="input_y")
    return X, y


def inference(X, embedded_W=None):
    """ Build the convolution model.

    Input
    - embedded_W: initialized value for embedded layer weights.
    Output
    - softmax: tensorflow variable for final softmax.
    """
    with tf.variable_scope('EmbeddedLayer'):
        W = tf.Variable(embedded_W, name="weights", dtype=tf.float32)
        embedded_X = tf.nn.embedding_lookup(W, X, name="X")
        # expand X to shape
        embedded_X_expand = tf.expand_dims(embedded_X, -1, name="X_expand")
        activation_summary_(embedded_X_expand)

    # doing multi-way convolution
    conv_outputs = []
    for filter_size in FILTER_SIZES:
        with tf.variable_scope('Conv-%d' % filter_size):
            filter_shape = [filter_size, EMBEDDING_SIZE, 1, NUM_FILTERS]
            W = variable_init_gaussian_with_weight_decay_(
                name="weights",
                shape=filter_shape,
                stddev=INIT_STDDEV,
                wd=WEIGHT_DECAY
            )
            b = variable_init_constant_("bias", [NUM_FILTERS], INIT_STDDEV)
            conv = tf.nn.conv2d(
                embedded_X_expand, W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv-%d" % filter_size
            )
            relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            activation_summary_(relu)

            pool = tf.nn.max_pool(
                relu,
                ksize=[1, SEQUENCE_LENGTH - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="pool"
            )
            conv_outputs.append(pool)

    with tf.variable_scope("Concat"):
        # concat and apply dropout
        num_filter_total = NUM_FILTERS * len(FILTER_SIZES)
        concat = tf.concat(3, conv_outputs, name="concat")
        concat_flat = tf.reshape(
            concat,
            [-1, num_filter_total],
            name="flat"
        )
        concat_flat_drop = tf.nn.dropout(
            concat_flat,
            DROPOUT_RATE,
            name="dropout"
        )
        activation_summary_(concat_flat_drop)

    with tf.variable_scope("FullyConnected"):
        W = variable_init_gaussian_with_weight_decay_(
            name="weights",
            shape=[num_filter_total, NUM_CLASSES],
            stddev=INIT_STDDEV
        )
        b = variable_init_constant_("bias", [NUM_CLASSES], 0)
        logits = tf.add(tf.matmul(concat_flat_drop, W), b, name="logits")
        activation_summary_(logits)

    return logits


def loss(logits, y):
    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, y, name="cross_entropy")
    cross_ent_mean = tf.reduce_mean(cross_ent, name="cross_entropy_mean")
    tf.add_to_collection('losses', cross_ent_mean)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return total_loss


def train(total_loss, global_step, num_iter_per_epoch):
    decay_steps = DECAY_EPOCH * num_iter_per_epoch
    lr = tf.train.exponential_decay(
        INIT_LEARNING_RATE,
        global_step,
        decay_steps,
        DECAY_FACTOR,
        staircase=True
    )
    tf.scalar_summary('learning rate', lr)

    moving_loss_op = moving_loss_(total_loss)

    # Compute gradients after loss computed.
    with tf.control_dependencies([moving_loss_op]):
        opt = tf.train.MomentumOptimizer(lr, MOMENTUM)
        grads = opt.compute_gradients(total_loss)

    # apply gradient update
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradient', grad)

    variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def accuracy(logits, y):
    correct = tf.nn.in_top_k(logits, y, 1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))
