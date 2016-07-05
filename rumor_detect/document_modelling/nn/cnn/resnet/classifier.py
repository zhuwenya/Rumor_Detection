# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import tensorflow as tf
from rumor_detect.document_modelling.nn.common.classifier import \
    NeuralNetworkClassifier


class CNNResnetClassifier(NeuralNetworkClassifier):
    """ Residual style convolutional neural networks.

    """
    def inference_(self, X, length, embedded_W, is_training=True):
        config = self.config_
        regularizer = tf.contrib.layers.l2_regularizer(config['WEIGHT_DECAY'])
        weight_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer()

        with tf.variable_scope("EmbeddingLayer"):
            embedded_X = tf.nn.embedding_lookup(embedded_W, X, name="X")
            embedded_X_reshape = tf.reshape(
                embedded_X,
                shape=[config['BATCH_SIZE'], config['SEQUENCE_MAX_LEN'], 1, -1]
            )

        with tf.variable_scope("Conv-1"):
            input_channel = int(embedded_X_reshape.get_shape()[3])
            W1 = tf.get_variable(
                name="weights",
                shape=[3, 1, input_channel, 16],
                initializer=weight_initializer,
                regularizer=regularizer
            )
            b1 = tf.get_variable(
                name="bias",
                shape=[16],
                initializer=bias_initializer
            )
            conv1 = tf.nn.bias_add(
                value=tf.nn.conv2d(
                    embedded_X_reshape,
                    W1,
                    [1, 1, 1, 1],
                    'SAME',
                    name='conv'
                ),
                bias=b1,
                name='conv_bias'
            )
            bn1 = self.spatial_batch_norm(conv1, is_training)
            relu1 = tf.nn.relu(bn1, name="relu")
            pool1 = tf.nn.max_pool(
                relu1,
                ksize=[1, 5, 1, 1],
                strides=[1, 5, 1, 1],
                padding="VALID",
                name="pool"
            )

        with tf.variable_scope("Conv-2"):
            conv2 = self.residual_block(pool1, 32, is_training)
            pool2 = tf.nn.max_pool(
                conv2,
                ksize=[1, 5, 1, 1],
                strides=[1, 5, 1, 1],
                padding="VALID",
                name="pool"
            )

        with tf.variable_scope("Conv-3"):
            conv3 = self.residual_block(pool2, 64, is_training)
            pool3 = tf.nn.max_pool(
                conv3,
                ksize=[1, 5, 1, 1],
                strides=[1, 5, 1, 1],
                padding="VALID",
                name="pool"
            )

        with tf.variable_scope("Conv-4"):
            conv3 = self.residual_block(pool3, 128, is_training)
            pool4 = tf.nn.max_pool(
                conv3,
                ksize=[1, 4, 1, 1],
                strides=[1, 4, 1, 1],
                padding="VALID",
                name="pool"
            )

        with tf.variable_scope("Conv-5"):
            conv5 = self.residual_block(pool4, 256, is_training)
            pool5 = tf.nn.avg_pool(
                conv5,
                ksize=[1, config['SEQUENCE_MAX_LEN'] / 500, 1, 1],
                strides=[1, config['SEQUENCE_MAX_LEN'] / 500, 1, 1],
                padding="VALID",
                name="pool"
            )

        with tf.variable_scope("FC-6"):
            flat = tf.reshape(pool5, [-1, 256])
            flat_dropout = tf.nn.dropout(
                flat,
                config['DROPOUT_KEEP_RATE'] if is_training else 1.0,
                name="dropout"
            )

            W = tf.get_variable(
                name="weights",
                shape=[256, config['NUM_CLASSES']],
                initializer=weight_initializer,
                regularizer=regularizer
            )
            b = tf.get_variable(
                name="bias",
                shape=[config['NUM_CLASSES']],
                initializer=bias_initializer
            )
            logits = tf.add(tf.matmul(flat_dropout, W), b, name="logits")

        return logits

    def residual_block(self, input, output_channel, is_training):
        config = self.config_
        regularizer = tf.contrib.layers.l2_regularizer(config['WEIGHT_DECAY'])
        weight_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer()

        input_channel = int(input.get_shape()[3])

        with tf.variable_scope('Residual-1'):
            W1 = tf.get_variable(
                name="weights",
                shape=[3, 1, input_channel, output_channel],
                initializer=weight_initializer,
                regularizer=regularizer
            )
            b1 = tf.get_variable(
                name="bias",
                shape=[output_channel],
                initializer=bias_initializer
            )
            conv1 = tf.nn.bias_add(
                value=tf.nn.conv2d(
                    input,
                    W1,
                    [1, 1, 1, 1],
                    'SAME',
                    name='conv'
                ),
                bias=b1,
                name='conv_bias'
            )
            bn1 = self.spatial_batch_norm(conv1, is_training)
            relu1 = tf.nn.relu(bn1, name="relu")

        with tf.variable_scope('Residual-2'):
            W2 = tf.get_variable(
                name="weights",
                shape=[3, 1, output_channel, output_channel],
                initializer=weight_initializer,
                regularizer=regularizer
            )
            b2 = tf.get_variable(
                name="bias",
                shape=[output_channel],
                initializer=bias_initializer
            )
            conv2 = tf.nn.bias_add(
                value=tf.nn.conv2d(
                    relu1,
                    W2,
                    [1, 1, 1, 1],
                    'SAME',
                    name='conv'
                ),
                bias=b2,
                name='conv_bias'
            )
            bn2 = self.spatial_batch_norm(conv2, is_training)

        with tf.variable_scope('Main'):
            remain = output_channel - input_channel
            padding = tf.pad(
                tensor=input,
                paddings=[[0, 0], [0, 0], [0, 0], [remain / 2, remain / 2]],
                name='padding'
            )
            main = self.spatial_batch_norm(padding, is_training)

        with tf.variable_scope('Add'):
            output = tf.add(main, bn2, name='output')
            output = tf.nn.relu(output, name='relu')

        return output

    def spatial_batch_norm(self, tensor, is_training):
        num_channel = int(tensor.get_shape()[3])
        config = self.config_

        with tf.variable_scope('BatchNorm'):
            gamma = tf.get_variable(
                name='gamma',
                shape=[num_channel],
                initializer=tf.constant_initializer(1)
            )
            beta = tf.get_variable(
                name='beta',
                shape=[num_channel],
                initializer=tf.constant_initializer()
            )
            moving_mean = tf.get_variable(
                name='moving_mean',
                shape=[num_channel],
                initializer=tf.constant_initializer(1),
                trainable=False
            )
            moving_var = tf.get_variable(
                name='moving_var',
                shape=[num_channel],
                initializer=tf.constant_initializer(),
                trainable=False
            )
            ema = tf.train.ExponentialMovingAverage(config['MOVING_AVG'])

            if is_training:
                mean, var = tf.nn.moments(tensor, [0, 1, 2], name='moments')
                moving_mean_op = moving_mean.assign(mean)
                moving_var_op = moving_var.assign(var)
                ema_op = ema.apply([moving_mean_op, moving_var_op])
                with tf.control_dependencies([ema_op]):
                    output = tf.nn.batch_normalization(
                        x=tensor,
                        mean=mean,
                        variance=var,
                        offset=beta,
                        scale=gamma,
                        variance_epsilon=1e-5,
                        name='output'
                    )
            else:
                output = tf.nn.batch_normalization(
                        x=tensor,
                        mean=moving_mean,
                        variance=moving_var,
                        offset=beta,
                        scale=gamma,
                        variance_epsilon=1e-5,
                        name='output'
                )
            return output


