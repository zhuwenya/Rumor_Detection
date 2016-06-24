# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import datetime
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

import time

from parameter import *
from rumor_detect.document_modelling.single_layer_cnn.batch_generator import \
    BatchGenerator
from rumor_detect.document_modelling.single_layer_cnn.model import \
    input_placeholder, inference, loss, train, accuracy
from rumor_detect.document_modelling.single_layer_cnn.preprocess.vocabulary import \
    Vocabulary


def run(train_batch_generator, dev_batch_generator, num_words):
    num_iter_per_epoch = train_batch_generator.num_iter_per_epoch()
    num_iter_total = num_iter_per_epoch * NUM_EPOCH

    with tf.Graph().as_default(),\
         tf.device('/gpu:1'):
        global_step = tf.Variable(0, trainable=False)

        # construct graph
        X, y = input_placeholder()
        logits = inference(X, num_words)
        total_loss = loss(logits, y)
        accuracy_op = accuracy(logits, y)
        train_op = train(total_loss, global_step, num_iter_per_epoch)

        # build the summary operation based on the graph.
        summary_op = tf.merge_all_summaries()

        # build the initialization operation.
        init_op = tf.initialize_all_variables()

        config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        )
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # initialize variables first
        sess.run(init_op)

        summary_writer = tf.train.SummaryWriter(LOG_DIR, sess.graph)

        for step in xrange(num_iter_total):
            start_time = time.time()
            X_feed, y_feed = train_batch_generator.next_batch()
            feed_dict = {X: X_feed, y: y_feed}
            _, loss_value = sess.run([train_op, total_loss],
                                     feed_dict=feed_dict)
            duration = time.time() - start_time

            if step % 10 == 0:
                msg = "%s: step=%d, loss=%f, batch_time=%.3f"
                print msg % (datetime.datetime.now(), step, loss_value, duration)

            if step % 20 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if (step % TEST_PER_ITER == 0 and step != 0) or \
               step == num_iter_total - 1:
                accuracy_values = []
                for i in xrange(TEST_NUM_BATCH):
                    X_feed, y_feed = dev_batch_generator.next_batch()
                    feed_dict = {X: X_feed, y: y_feed}
                    accuracy_value = sess.run([accuracy_op], feed_dict=feed_dict)
                    accuracy_values.append(accuracy_value)
                dev_accuracy = np.mean(accuracy_values)
                msg = "%s: step=%d, dev_accuracy=%.3f"
                print msg % (datetime.datetime.now(), step, dev_accuracy)


if __name__ == "__main__":
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
        "vocab_path",
        help="vocabulary data location"
    )
    args = parser.parse_args()

    vocab = Vocabulary.load(args.vocab_path)
    train_generator = BatchGenerator(args.train_path, vocab)
    dev_generator = BatchGenerator(args.dev_path, vocab)
    run(train_generator, dev_generator, vocab.number_words())
