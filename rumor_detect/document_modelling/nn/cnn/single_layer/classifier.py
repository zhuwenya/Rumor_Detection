# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import tensorflow as tf
from rumor_detect.document_modelling.nn.common.classifier import \
    NeuralNetworkClassifier


class CNNSingleLayerClassifier(NeuralNetworkClassifier):
    """ Single convolution layer classifier based on
    [Yoon Kim. 2014. Convolutional Neural Networks for Sentence Classification.]
    """
    def inference_(self, X, length, embedded_W, is_training=True):
        config = self.config_
        with tf.variable_scope('EmbeddedLayer'):
            embedded_X = tf.nn.embedding_lookup(embedded_W, X, name="X")
            embedded_X_expand = tf.expand_dims(embedded_X, -1, name="X_expand")

        initializer = tf.truncated_normal_initializer(
            stddev=config['GAUSSIAN_STDDEV']
        )
        regularizer = tf.contrib.layers.l2_regularizer(config['WEIGHT_DECAY'])

        # doing multi-way convolution
        conv_outputs = []
        for filter_size in config['FILTER_SIZES']:
            with tf.variable_scope('Conv-%d' % filter_size):
                embedded_size = embedded_W.get_shape()[1]
                filter_shape = [filter_size, embedded_size,
                                1, config['NUM_FILTERS']]
                W = tf.get_variable(
                    name="weights",
                    shape=filter_shape,
                    initializer=initializer,
                    regularizer=regularizer
                )
                b = tf.get_variable(
                    name="bias",
                    shape=[config['NUM_FILTERS']],
                    initializer=tf.constant_initializer(0)
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
                    ksize=[1, config['SEQUENCE_MAX_LEN']-filter_size+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool"
                )
                conv_outputs.append(pool)

        with tf.variable_scope("Concat"):
            # concat and apply dropout
            num_filter_total = config['NUM_FILTERS']*len(config['FILTER_SIZES'])
            concat = tf.concat(3, conv_outputs, name="concat")
            concat_flat = tf.reshape(
                concat,
                [-1, num_filter_total],
                name="flat"
            )
            concat_flat_drop = tf.nn.dropout(
                concat_flat,
                config['DROPOUT_KEEP_RATE'] if is_training else 1.0,
                name="dropout"
            )

        with tf.variable_scope("FullyConnected"):
            W = tf.get_variable(
                name="weights",
                shape=[num_filter_total, config['NUM_CLASSES']],
                initializer=initializer,
                regularizer=regularizer
            )
            b = tf.get_variable(
                name="bias",
                shape=[config['NUM_CLASSES']],
                initializer=tf.constant_initializer()
            )
            logits = tf.add(tf.matmul(concat_flat_drop, W), b, name="logits")

        return logits
