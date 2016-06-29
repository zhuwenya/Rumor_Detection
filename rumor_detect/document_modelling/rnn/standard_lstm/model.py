# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import tensorflow as tf
from parameters import *


def initialize_embedding_matrix(W):
    return tf.Variable(W, name="embedded_weights", dtype=tf.float32)


def create_placeholder():
    X = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQUENCE_MAX_LENGTH], name="input_X")
    y = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name="input_y")
    length = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name="input_length")
    return X, y, length


def last_relevant(outputs, length):
    index = tf.range(0, BATCH_SIZE) * SEQUENCE_MAX_LENGTH + (length - 1)
    flat = tf.reshape(outputs, [-1, RNN_HIDDEN_SIZE])
    relevant = tf.gather(flat, index, name='last')
    return relevant


def inference(X, length, embedded_W, is_training=True):
    embedded_X = tf.nn.embedding_lookup(embedded_W, X, name="X")

    # create lstm
    lstm_cell = tf.nn.rnn_cell.LSTMCell(
        num_units=RNN_HIDDEN_SIZE,
        state_is_tuple=True,
        initializer=tf.truncated_normal_initializer(stddev=NORMAL_INIT_STDDEV)
    )
    if is_training:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=lstm_cell,
            output_keep_prob=DROPOUT_KEEP_PROB
        )
    outputs, _ = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=embedded_X,
        sequence_length=length,
        dtype=tf.float32
    )
    last = last_relevant(outputs, length)

    # fully connected layer
    W = tf.get_variable(
        name="FC_weights",
        shape=[RNN_HIDDEN_SIZE, NUM_CLASSES],
        initializer=tf.truncated_normal_initializer(stddev=NORMAL_INIT_STDDEV),
        regularizer=tf.contrib.layers.l2_regularizer(L2_WEIGHT_DECAY)
    )
    b = tf.get_variable(
        name="FC_bias",
        shape=[NUM_CLASSES],
        initializer=tf.constant_initializer()
    )

    logits = tf.add(tf.matmul(last, W), b, name="logits")
    tf.histogram_summary('logits', logits)

    return logits


def loss(logits, y):
    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=y
    )
    cross_ent_mean = tf.reduce_mean(cross_ent, name='empirical_loss')
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_op = tf.add_n(reg_losses + [cross_ent_mean], "loss")

    tf.scalar_summary('empirical loss', cross_ent_mean)
    tf.scalar_summary('loss', loss_op)

    return loss_op


def accuracy(logits, y):
    correct = tf.nn.in_top_k(logits, y, 1)
    correct_mean = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.scalar_summary('accuracy', correct_mean)
    return correct_mean
