# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

# data
BATCH_SIZE = 128
SEQUENCE_MAX_LENGTH = 2000
NUM_CLASSES = 2

# model
NUM_FILTERS = 100
FILTER_SIZES = [2, 3, 4, 5]
DROPOUT_KEEP_PROB = .5
INIT_STDDEV = 1e-2
WEIGHT_DECAY = 1e-5

# train
INIT_LEARNING_RATE = 1e-2
NUM_TRAIN_EPOCH = 60
NUM_TRAIN_EPOCH_DECAY = 15
DECAY_FACTOR = 0.1
NUM_TRAIN_EPOCH_TEST = 2

# directory
LOG_DIR = "/tmp/tensorflow/cnn/log/"
SAVE_DIR = "/tmp/tensorflow/cnn/model/"
