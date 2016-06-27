# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

# data
BATCH_NUM = 128
SEQUENCE_LENGTH = 2000
NUM_CLASSES = 2

# model
EMBEDDING_SIZE = 100
NUM_FILTERS = 100
FILTER_SIZES = [2, 3, 4, 5]
DROPOUT_KEEP_PROB = .5
INIT_STDDEV = 1e-2
WEIGHT_DECAY = 1e-5
MOVING_AVG = 0.999

# train
INIT_LEARNING_RATE = 1e-2
NUM_EPOCH = 60
DECAY_EPOCH = 20
DECAY_FACTOR = 0.1
MOMENTUM = 0.9
TEST_PER_ITER = 500
TEST_NUM_BATCH = 50

# directory
LOG_DIR = "/tmp/tensorflow_log/"
SAVE_DIR = "/tmp/tensorflow_save/"
