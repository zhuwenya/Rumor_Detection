# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
from argparse import ArgumentParser
import logging

import numpy as np
from sklearn.metrics import precision_recall_curve
import tensorflow as tf
from parameter import *
from rumor_detect.document_modelling.cnn.model import \
    input_placeholder, inference, initialize_embedding_matrix
from rumor_detect.document_modelling.utils.rumor_corpus_for_nn import \
    RumorCorpusForNN
from rumor_detect.document_modelling.utils.tf_train_utils import predict
from rumor_detect.document_modelling.utils.word2vec_lookup_table import \
    Word2VecLookupTable

logger = logging.getLogger('cnn.eval')

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
        "word2vec_path",
        help="word2vec model file path"
    )
    args = parser.parse_args()

    word2vec_lookup_table = Word2VecLookupTable(args.word2vec_path)
    valid_corpus = RumorCorpusForNN(
        path=args.dev_path,
        word2vec_lookup_table=word2vec_lookup_table,
        fixed_size=SEQUENCE_MAX_LENGTH
    )

    with tf.Graph().as_default():
        # create operation
        W = word2vec_lookup_table.embedding_matrix()
        embedded_W_cpu = word2vec_lookup_table.embedding_matrix()
        embedded_W_gpu = initialize_embedding_matrix(embedded_W_cpu)

        X, y = input_placeholder()
        logits = inference(X, embedded_W_gpu, is_training=False)
        y_pred_op = predict(X)

        # restore the moving average version of the learned variables for eval
        variable_averages = tf.train.ExponentialMovingAverage(0.999)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        logger.info('restoring variables...')
        sess = tf.Session()

        # load file to restore variables
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        saver.restore(sess, ckpt.model_checkpoint_path)

        logger.info('validating (moving average and dropout applied)...')

        y_pred_batches, y_gt_batches = [], []
        num_test_iter = valid_corpus.num_iter_in_epoch(BATCH_SIZE)
        for i in xrange(num_test_iter):
            X_feed, y_feed, _ = valid_corpus.next_batch(BATCH_SIZE)
            feed_dict = {X: X_feed, y: y_feed}
            y_pred_batch = sess.run([y_pred_op], feed_dict=feed_dict)
            y_pred_batches.append(y_pred_batch)
            y_gt_batches.append(y_feed)

        y_pred = np.vstack(y_pred_batches)
        y_gt = np.vstack(y_gt_batches).flatten()
        precison, recall, thresholds = precision_recall_curve(y_gt, y_pred[:, NUM_CLASSES-1])
        print precison
        print recall
        print thresholds