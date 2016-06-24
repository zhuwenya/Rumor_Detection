# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

# data
BATCH_NUM = 64
SEQUENCE_LENGTH = 1000
NUM_CLASSES = 2

# model
EMBEDDING_SIZE = 100
NUM_FILTERS = 100
FILTER_SIZES = [2, 3, 4, 5]
DROPOUT_RATE = .5
INIT_STDDEV = 1e-2
WEIGHT_DECAY = None

# train
INIT_LEARNING_RATE = 1e-2
NUM_EPOCH = 60
DECAY_EPOCH = 20
DECAY_FACTOR = 0.1
MOMENTUM = 0.9
TEST_PER_ITER = 100
TEST_NUM_BATCH = 15

# log directory
LOG_DIR = "/tmp/tensorflow"