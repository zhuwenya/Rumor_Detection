# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import tensorflow as tf
from rumor_detect.document_modelling.nn.common.classifier \
    import NeuralNetworkClassifier


class SoftmaxClassifier(NeuralNetworkClassifier):
    """ Vanilla softmax classifier based on lookup table.

    This classifier first embedded each word into word vector.
    A softmax layer is placed upon those word vectors to perform classifying.
    """
    def inference_(self, X, length, embedded_W, is_training=True):
        config = self.config_
        embedded_X = tf.nn.embedding_lookup(embedded_W, X, name='X')
        flat_X = tf.reshape(
            embedded_X,
            shape=[config['BATCH_SIZE'], -1],
            name='X_flat'
        )
        W = tf.get_variable(
            name="softmax_weights",
            shape=[flat_X.get_shape()[1], config['NUM_CLASSES']],
            initializer=tf.truncated_normal_initializer(stddev=0.01)
        )
        logits = tf.matmul(flat_X, W, name="logits")
        return logits