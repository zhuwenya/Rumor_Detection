# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import logging
import os
import time
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

from parameter import *
from rumor_detect.document_modelling.cnn.model import \
    input_placeholder, inference, initialize_embedding_matrix
from rumor_detect.document_modelling.utils.rumor_corpus_for_nn import \
    RumorCorpusForNN
from rumor_detect.document_modelling.utils.tf_train_utils import train, loss, \
    accuracy
from rumor_detect.document_modelling.utils.word2vec_lookup_table import \
    Word2VecLookupTable

logger = logging.getLogger("cnn.train")


def run(train_corpus, valid_corpus, word2vec_lookup_table):
    num_train_iter_per_epoch = train_corpus.num_iter_in_epoch(BATCH_SIZE)
    num_train_iter_per_decay = num_train_iter_per_epoch * NUM_TRAIN_EPOCH_DECAY
    num_train_iter_per_test = num_train_iter_per_epoch * NUM_TRAIN_EPOCH_TEST
    num_train_iter_total = num_train_iter_per_epoch * NUM_TRAIN_EPOCH

    with tf.Graph().as_default():
        logger.info('loading embedding matrix...')
        embedded_W_cpu = word2vec_lookup_table.embedding_matrix()
        embedded_W_gpu = initialize_embedding_matrix(embedded_W_cpu)

        # construct graph
        X, y = input_placeholder()
        logits = inference(X, embedded_W_gpu, is_training=True)
        loss_op = loss(logits, y)
        accuracy_op = accuracy(logits, y)
        global_step = tf.Variable(0, trainable=False)
        train_op = train(
            loss=loss_op,
            global_step=global_step,
            decay_steps=num_train_iter_per_decay,
            initial_learning_rate=INIT_LEARNING_RATE,
            decay_rate=DECAY_FACTOR
        )

        config = tf.ConfigProto(
            allow_soft_placement=True
        )
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # initialize variables first
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(LOG_DIR, sess.graph)

        for step in xrange(num_train_iter_total):
            start_time = time.time()
            X_feed, y_feed, _ = train_corpus.next_batch(BATCH_SIZE)
            feed_dict = {X: X_feed, y: y_feed}
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

            if (step % num_train_iter_per_test == 0 and step != 0) or \
               step + 1 == num_train_iter_total:
                accuracy_values = []
                num_test_iter = valid_corpus.num_iter_in_epoch(BATCH_SIZE)
                for i in xrange(num_test_iter):
                    X_feed, y_feed, _ = valid_corpus.next_batch(BATCH_SIZE)
                    feed_dict = {X: X_feed, y: y_feed}
                    accuracy_value = sess.run(
                        accuracy_op,
                        feed_dict=feed_dict
                    )
                    accuracy_values.append(accuracy_value)
                valid_accuracy = np.mean(accuracy_values)
                msg = "validation step=%d, dev_accuracy=%.3f"
                logger.info(msg % (step, valid_accuracy))

        # saving models
        logger.info('saving model...')
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        saver = tf.train.Saver(tf.all_variables())
        save_path = os.path.join(SAVE_DIR, 'model.ckpt')
        saver.save(sess, save_path, global_step=num_train_iter_total-1)


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
