# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
from argparse import ArgumentParser
import logging

import numpy as np
import tensorflow as tf
from parameter import *
from rumor_detect.document_modelling.single_layer_cnn.batch_generator import \
    BatchGenerator
from rumor_detect.document_modelling.single_layer_cnn.model import \
    input_placeholder, inference, accuracy
from rumor_detect.document_modelling.single_layer_cnn.preprocess.vocabulary import \
    Vocabulary
from rumor_detect.document_modelling.single_layer_cnn.train import \
    init_embedded_weights

logger = logging.getLogger('eval.py')

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    parser = ArgumentParser()
    parser.add_argument(
        "dev_path",
        help="validation data location."
    )
    parser.add_argument(
        "vocab_path",
        help="vocabulary data location"
    )
    parser.add_argument(
        "word2vec_path",
        help="model file for word2vec location."
    )
    args = parser.parse_args()

    logger.info('loading vocabulary...')
    vocab = Vocabulary.load(args.vocab_path)
    logger.info('initializing weights...')
    W = init_embedded_weights(vocab, args.word2vec_path)
    dev_generator = BatchGenerator(args.dev_path, vocab)

    with tf.Graph().as_default():
        # create operation
        X, y = input_placeholder()
        logits = inference(X, W, is_train=False)
        accuracy_op = accuracy(logits, y)

        # restore the moving average version of the learned variables for eval
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVG)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        logger.info('restoring variables...')
        sess = tf.Session()

        # load file to restore variables
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        saver.restore(sess, ckpt.model_checkpoint_path)

        logger.info('validating (moving average and dropout applied)...')
        accuracy_values = []
        for i in xrange(TEST_NUM_BATCH):
            X_feed, y_feed = dev_generator.next_batch()
            feed_dict = {X: X_feed, y: y_feed}
            accuracy_value = sess.run([accuracy_op], feed_dict=feed_dict)
            accuracy_values.append(accuracy_value)
        dev_accuracy = np.mean(accuracy_values)
        logger.info('validation accuracy=%.3f' % dev_accuracy)