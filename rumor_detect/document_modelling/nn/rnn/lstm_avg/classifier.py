# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import tensorflow as tf
from rumor_detect.document_modelling.nn.common.classifier import \
    NeuralNetworkClassifier


class LSTMAverageClassifier(NeuralNetworkClassifier):
    """ Variant of LSTM.

    This model use LSTM to get a good representation of sequences. The hidden
    vectors are averaged to get a sequence encoded vector. To do classification,
    a softmax layer is placed upon the encoded vector.
    """
    def inference_(self, X, length, embedded_W, is_training=True):
        config = self.config_

        mask = X != self.PADDING_IDX_
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
        avg = self.average_(outputs, mask, length)

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

        logits = tf.add(tf.matmul(avg, W), b, name="logits")
        tf.histogram_summary('logits', logits)

        return logits

    def average_(self, outputs, mask, length):
        sum = tf.reduce_sum(outputs*mask, reduction_indices=1, name='lstm_sum')
        length_for_div = tf.expand_dims(tf.cast(length, dtype=tf.float32), -1)
        avg = tf.div(sum, length_for_div, name='lstm_avg')
        return avg

    def run_train(self, train_corpus, valid_corpus, word2vec_lookup_table):
        self.PADDING_IDX_ = word2vec_lookup_table.get_padding_word_idx()
        NeuralNetworkClassifier.run_train(
            self,
            train_corpus,
            valid_corpus,
            word2vec_lookup_table
        )

    def run_eval(self, valid_corpus, word2vec_lookup_table):
        self.PADDING_IDX_ = word2vec_lookup_table.get_padding_word_idx()
        NeuralNetworkClassifier.run_eval(
            self,
            valid_corpus,
            word2vec_lookup_table
        )
