# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import tensorflow as tf
from parameter import *


def initialize_embedding_matrix(W):
    return tf.Variable(W, name="embedded_weights", dtype=tf.float32)


def input_placeholder():
    X = tf.placeholder(
        dtype=tf.int32,
        shape=[None, SEQUENCE_MAX_LENGTH],
        name="input_X"
    )
    y = tf.placeholder(tf.int32, shape=[None], name="input_y")
    return X, y


def inference(X, embedded_W, is_training=True):
    """ Build the convolution model.

    Input
    - embedded_W: initialized value for embedded layer weights.
    - is_training: boolean value. If set to true, then training policy will
      applied. Otherwise it will apply testing policy.
    Output
    - logits
    """
    with tf.variable_scope('EmbeddedLayer'):
        embedded_X = tf.nn.embedding_lookup(embedded_W, X, name="X")
        embedded_X_expand = tf.expand_dims(embedded_X, -1, name="X_expand")

    # doing multi-way convolution
    conv_outputs = []
    for filter_size in FILTER_SIZES:
        with tf.variable_scope('Conv-%d' % filter_size):
            embedded_size = embedded_W.get_shape()[1]
            filter_shape = [filter_size, embedded_size, 1, NUM_FILTERS]
            W = tf.get_variable(
                name="weights",
                shape=filter_shape,
                initializer=tf.truncated_normal_initializer(stddev=INIT_STDDEV),
                regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
            )
            b = tf.get_variable(
                name="bias",
                shape=[NUM_FILTERS],
                initializer=tf.constant_initializer(INIT_STDDEV)
            )
            conv = tf.nn.conv2d(
                embedded_X_expand, W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv-%d" % filter_size
            )
            relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            pool = tf.nn.max_pool(
                relu,
                ksize=[1, SEQUENCE_MAX_LENGTH - filter_size + 1, 1, 1],
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
            DROPOUT_KEEP_PROB if is_training else 1.0,
            name="dropout"
        )

    with tf.variable_scope("FullyConnected"):
        W = tf.get_variable(
            name="weights",
            shape=[num_filter_total, NUM_CLASSES],
            initializer=tf.truncated_normal_initializer(stddev=INIT_STDDEV),
            regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        )
        b = tf.get_variable(
            name="bias",
            shape=[NUM_CLASSES],
            initializer=tf.constant_initializer()
        )
        logits = tf.add(tf.matmul(concat_flat_drop, W), b, name="logits")

    return logits

