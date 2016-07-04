# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import tensorflow as tf
from rumor_detect.document_modelling.nn.common.classifier import \
    NeuralNetworkClassifier


class LSTMStandardClassifier(NeuralNetworkClassifier):
    """ Vanllia implementation of LSTM.

    This model use LSTM to get a good representation of sequences. The last
    hidden vector is sequence encoded vector. To do classification, a softmax
    layer is placed upon the encoded vector.
    """
    def inference_(self, X, length, embedded_W, is_training=True):
        config = self.config_

        embedded_X = tf.nn.embedding_lookup(embedded_W, X, name="X")
        initializer = tf.truncated_normal_initializer(
            stddev=config['GAUSSIAN_STDDEV']
        )
        regularizer = tf.contrib.layers.l2_regularizer(config['WEIGHT_DECAY'])

        # create lstm
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=config['RNN_HIDDEN_SIZE'],
            state_is_tuple=True,
            initializer=initializer
        )
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=lstm_cell,
                output_keep_prob=config['DROPOUT_KEEP_RATE']
            )
        outputs, _ = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=embedded_X,
            sequence_length=length,
            dtype=tf.float32
        )
        last = self.last_relevant_(outputs, length)

        # fully connected layer
        W = tf.get_variable(
            name="FC_weights",
            shape=[config['RNN_HIDDEN_SIZE'], config['NUM_CLASSES']],
            initializer=initializer,
            regularizer=regularizer
        )
        b = tf.get_variable(
            name="FC_bias",
            shape=[config['NUM_CLASSES']],
            initializer=tf.constant_initializer()
        )

        logits = tf.add(tf.matmul(last, W), b, name="logits")
        tf.histogram_summary('logits', logits)

        return logits

    def last_relevant_(self, outputs, length):
        config = self.config_
        index = tf.range(0, config['BATCH_SIZE']) * config['SEQUENCE_MAX_LEN'] \
                + (length - 1)
        flat = tf.reshape(outputs, [-1, config['RNN_HIDDEN_SIZE']])
        relevant = tf.gather(flat, index, name='last')
        return relevant

