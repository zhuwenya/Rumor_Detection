# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

SEQUENCE_MAX_LENGTH = 2000

BATCH_SIZE = 64
NUM_CLASSES = 2
RNN_HIDDEN_SIZE = 100
DROPOUT_KEEP_PROB = 0.5

NORMAL_INIT_STDDEV = 0.01
L2_WEIGHT_DECAY = 1e-5

NUM_EPOCH = 60
NUM_DECAY_EPOCH = 20
LEARNING_RATE_INIT = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.1
TEST_AFTER_TRAIN_EPOCH = 10

LOG_DIR = '/tmp/tensorflow/rnn/log/'
MODEL_DIR = '/tmp/tensorflow/rnn/model/'
