# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
from argparse import ArgumentParser
import logging
import os
import time

import numpy as np
import tensorflow as tf

from parameters import *
from rumor_detect.document_modelling.utils.tf_train_utils import train, loss, \
    accuracy
from rumor_detect.document_modelling.nn.common.rumor_corpus_for_nn import \
    RumorCorpusForNN
from rumor_detect.document_modelling.rnn.standard_lstm.model import \
    create_placeholder, initialize_embedding_matrix, inference
from rumor_detect.document_modelling.nn.common.word2vec_lookup_table import \
    Word2VecLookupTable

logger = logging.getLogger('rnn.lstm_standard.train')


def run(train_corpus, valid_corpus, word2vec_lookup_table):
    num_iter_per_epoch = train_corpus.num_iter_in_epoch(BATCH_SIZE)
    num_iter_lr_decay = num_iter_per_epoch * NUM_DECAY_EPOCH
    num_iter = num_iter_per_epoch * NUM_EPOCH
    num_iter_invoke_test = num_iter_per_epoch * TEST_AFTER_TRAIN_EPOCH
    num_iter_in_test = valid_corpus.num_iter_in_epoch(BATCH_SIZE)

    with tf.Graph().as_default():
        # init embedding matrix
        logger.info('loading embedding matrix from word2vec.model...')
        embedded_W_cpu = word2vec_lookup_table.embedding_matrix()
        embedded_W_gpu = initialize_embedding_matrix(embedded_W_cpu)

        logger.info('creating graph...')
        global_step = tf.Variable(0, trainable=False)
        X, y, length = create_placeholder()
        logits = inference(X, length, embedded_W_gpu, is_training=True)
        loss_op = loss(logits, y)
        accuracy_op = accuracy(logits, y)
        train_op = train(
            loss=loss_op,
            global_step=global_step,
            decay_steps=num_iter_lr_decay,
            initial_learning_rate=LEARNING_RATE_INIT,
            decay_rate=LEARNING_RATE_DECAY_FACTOR,
            max_gradient_norm=MAX_GRADIENT_NORM
        )
        init_op = tf.initialize_all_variables()

        # create a saver
        saver = tf.train.Saver(tf.all_variables())

        # create a session
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        )
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # initialize variable first
        sess.run(init_op)

        # summary operation
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(LOG_DIR, sess.graph)

        logger.info('start backprop...')
        for step in xrange(num_iter):
            start_time = time.time()
            X_feed, y_feed, length_feed = train_corpus.next_batch(BATCH_SIZE)
            feed_dict = {X: X_feed, y: y_feed, length: length_feed}
            _, loss_val, acc_val = sess.run(
                [train_op, loss_op, accuracy_op],
                feed_dict=feed_dict
            )
            duration = time.time() - start_time

            if step % 10 == 0:
                msg = "step=%d, loss=%f, accuracy=%.3f, batch_time=%.3f"
                logger.info(msg % (step, loss_val, acc_val, duration))

            if step % 20 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if (step % num_iter_invoke_test == 0 and step != 0) or\
                step + 1 == num_iter:
                acc_vals = []
                for i in xrange(num_iter_in_test):
                    X_feed, y_feed, length_feed = \
                        valid_corpus.next_batch(BATCH_SIZE)
                    feed_dict = {X: X_feed, y: y_feed, length: length_feed}
                    acc_val = sess.run(accuracy_op, feed_dict=feed_dict)
                    acc_vals.append(acc_val)
                dev_acc = np.mean(acc_vals)
                logger.info('validation accuracy=%.3f' % dev_acc)

        logger.info('saving model...')
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        save_path = os.path.join(MODEL_DIR, 'model.ckpt')
        saver.save(sess, save_path=save_path, global_step=num_iter-1)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s]: %(message)s",
        level=logging.INFO
    )

    parser = ArgumentParser()
    parser.add_argument(
        "train_path",
        help="Training data location."
    )
    parser.add_argument(
        "dev_path",
        help="validation data location."
    )
    parser.add_argument(
        "word2vec_path",
        help="model file for word2vec location."
    )
    args = parser.parse_args()

    logger.info('building vocabulary from word2vec model file')
    word2vec_lookup_table = Word2VecLookupTable(args.word2vec_path)
    logger.info('loading train corpus...')
    train_corpus = RumorCorpusForNN(
        path=args.train_path,
        word2vec_lookup_table=word2vec_lookup_table,
        fixed_size=SEQUENCE_MAX_LENGTH
    )
    logger.info('loading validation corpus...')
    dev_corpus = RumorCorpusForNN(
        path=args.dev_path,
        word2vec_lookup_table=word2vec_lookup_table,
        fixed_size=SEQUENCE_MAX_LENGTH
    )
    logger.info('start training...')
    run(train_corpus, dev_corpus, word2vec_lookup_table)


