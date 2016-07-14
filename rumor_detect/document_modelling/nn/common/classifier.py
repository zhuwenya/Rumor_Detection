# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import logging
import os
import pickle
import numpy as np
import tensorflow as tf
import time
from rumor_detect.document_modelling.utils.metrics import print_metrics

logger = logging.getLogger('nn.classifier')


class NeuralNetworkClassifier(object):
    def __init__(self, config):
        self.config_ = config

    def run_train(self, train_corpus, valid_corpus, word2vec_lookup_table):
        config = self.config_
        train_steps_per_epoch = \
            train_corpus.num_iter_in_epoch(config['BATCH_SIZE'])
        decay_steps = \
            train_steps_per_epoch * config['NUM_DECAY_EPOCH']
        train_steps = \
            train_steps_per_epoch * config['NUM_TRAIN_EPOCH']
        test_steps = \
            train_steps_per_epoch * config['NUM_TRAIN_EPOCH_PER_TEST']

        with tf.Graph().as_default():
            # init embedding matrix
            logger.info('loading embedding matrix from word2vec.model...')
            embedded_W_cpu = word2vec_lookup_table.embedding_matrix()
            embedded_W_gpu = self.initialize_embedding_matrix_(embedded_W_cpu)

            logger.info('creating graph...')
            X, y, length = self.create_placeholder_()
            logits = \
                self.inference_(X, length, embedded_W_gpu, is_training=True)
            loss_op = self.loss_(logits, y)
            accuracy_op = self.accuracy_(logits, y)
            train_op = self.train_(loss=loss_op, decay_steps=decay_steps)
            init_op = tf.initialize_all_variables()

            # create a saver
            saver = tf.train.Saver(tf.all_variables())

            # create a session
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True
            sess = tf.Session(config=sess_config)

            # initialize variable first
            sess.run(init_op)

            # delete embedded_W_cpu to save memory
            del embedded_W_cpu

            # summary operation
            summary_op = tf.merge_all_summaries()
            summary_writer = \
                tf.train.SummaryWriter(config['LOG_DIR'], sess.graph)

            logger.info('start backprop...')
            for step in xrange(train_steps):
                start_time = time.time()
                X_feed, y_feed, length_feed = \
                    train_corpus.next_batch(config['BATCH_SIZE'])
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

                if (step + 1) % test_steps == 0 or step + 1 == train_steps:
                    acc_vals = []
                    num_iter_in_test = \
                        valid_corpus.num_iter_in_epoch(config['BATCH_SIZE'])
                    for i in xrange(num_iter_in_test):
                        X_feed, y_feed, length_feed = \
                            valid_corpus.next_batch(config['BATCH_SIZE'])
                        feed_dict = {X: X_feed, y: y_feed, length: length_feed}
                        acc_val = sess.run(accuracy_op, feed_dict=feed_dict)
                        acc_vals.append(acc_val)
                    valid_acc = np.mean(acc_vals)
                    logger.info('validation accuracy=%.3f' % valid_acc)

            # save model in the last iteration
            logger.info('saving model...')
            if not os.path.exists(config['MODEL_DIR']):
                os.makedirs(config['MODEL_DIR'])
            save_path = os.path.join(config['MODEL_DIR'], 'model.ckpt')
            saver.save(sess, save_path=save_path, global_step=train_steps - 1)

    def run_eval(self, valid_corpus, word2vec_lookup_table):
        config = self.config_

        with tf.Graph().as_default():
            logger.info('loading embedding matrix from word2vec.model...')
            embedded_W_cpu = word2vec_lookup_table.embedding_matrix()
            embedded_W_gpu = self.initialize_embedding_matrix_(embedded_W_cpu)

            X, y, length = self.create_placeholder_()
            logits = self.inference_(
                X=X,
                length=length,
                embedded_W=embedded_W_gpu,
                is_training=False
            )
            y_pred_op = self.predict_(logits)

            # restore the moving average version of the variables for eval
            variable_averages = \
                tf.train.ExponentialMovingAverage(config['MOVING_AVG'])
            variable_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variable_to_restore)

            sess = tf.Session()

            logger.info('restoring variables...')
            ckpt = tf.train.get_checkpoint_state(config['MODEL_DIR'])
            saver.restore(sess, ckpt.model_checkpoint_path)

            # delete embedded_W_cpu to save memory
            del embedded_W_cpu

            logger.info('validating...')

            num_test_iter = valid_corpus.num_iter_in_epoch(config['BATCH_SIZE'])
            y_pred = np.zeros(
                [num_test_iter * config['BATCH_SIZE'], config['NUM_CLASSES']]
            )
            y_gt = np.zeros(
                (num_test_iter * config['BATCH_SIZE'],),
                dtype=np.int32
            )

            for i in xrange(num_test_iter):
                X_feed, y_feed, length_feed = \
                    valid_corpus.next_batch(config['BATCH_SIZE'])
                feed_dict = {X: X_feed, y: y_feed, length: length_feed}
                y_pred_batch = sess.run(y_pred_op, feed_dict=feed_dict)

                start_idx = i * config['BATCH_SIZE']
                end_idx = start_idx + config['BATCH_SIZE']
                y_pred[start_idx:end_idx, :] = y_pred_batch
                y_gt[start_idx:end_idx] = y_feed

            logger.info('Classification report:')
            y_pred_label = np.round(y_pred[:, config['NUM_CLASSES'] - 1])
            print_metrics(y_gt, y_pred_label)

            # dump results
            logger.info('dump results...')
            m = {
                'y_true': y_gt,
                'y_pred': y_pred[:, config['NUM_CLASSES'] - 1]
            }

            if not os.path.exists(config['RESULT_DIR']):
                os.makedirs(config['RESULT_DIR'])
            save_path = os.path.join(config['RESULT_DIR'], 'result.pk')
            with open(save_path, 'w') as out_f:
                pickle.dump(m, out_f)

    def initialize_embedding_matrix_(self, W):
        return tf.Variable(W, name="embedded_weights", dtype=tf.float32)

    def create_placeholder_(self):
        config = self.config_
        X = tf.placeholder(
            dtype=tf.int32,
            shape=[config['BATCH_SIZE'], config['SEQUENCE_MAX_LEN']],
            name="input_X"
        )
        y = tf.placeholder(
            dtype=tf.int32,
            shape=[config['BATCH_SIZE']],
            name="input_y"
        )
        length = tf.placeholder(
            dtype=tf.int32,
            shape=[config['BATCH_SIZE']],
            name="input_length"
        )
        return X, y, length

    def inference_(self, X, length, embedded_W, is_training=True):
        config = self.config_
        logits = tf.get_variable(
            name="logits",
            shape=[config['BATCH_SIZE'], config['NUM_CLASSES']],
            initializer=tf.constant_initializer()
        )
        return logits

    def loss_(self, logits, y):
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=y
        )
        cross_ent_mean = tf.reduce_mean(cross_ent, name='empirical_loss')
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_op = tf.add_n(reg_losses + [cross_ent_mean], "loss")

        tf.scalar_summary('empirical loss', cross_ent_mean)
        tf.scalar_summary('loss', loss_op)

        return loss_op

    def accuracy_(self, logits, y):
        correct = tf.nn.in_top_k(logits, y, 1)
        correct_mean = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.scalar_summary('accuracy', correct_mean)
        return correct_mean

    def predict_(self, logits):
        return tf.nn.softmax(logits)

    def train_(self, loss, decay_steps):
        config = self.config_

        # create learning rate
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(
            learning_rate=config['INIT_LEARNING_RATE'],
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=config['DECAY_FACTOR'],
            staircase=True
        )
        tf.scalar_summary('learning rate', lr)

        # compute and apply gradients
        opt = tf.train.MomentumOptimizer(lr, config['MOMENTUM'])
        gradvars = opt.compute_gradients(loss)
        if config.get('MAX_GRAD_NORM', None):
            gradvars = [(tf.clip_by_norm(gv[0], config["MAX_GRAD_NORM"]), gv[1])
                        for gv in gradvars]
        apply_grads = opt.apply_gradients(gradvars, global_step)

        # add histograms for variables and gradients
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        for grad, var in gradvars:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradient', grad)

        # smoothing loss and variables
        ema = tf.train.ExponentialMovingAverage(config['MOVING_AVG'],
                                                global_step)
        ema_op = ema.apply(tf.trainable_variables() + [loss])
        tf.scalar_summary('moving_loss', ema.average(loss))

        # combine all op into a train_op
        with tf.control_dependencies([apply_grads]):
            train_op = tf.group(ema_op, name='train')

        return train_op
