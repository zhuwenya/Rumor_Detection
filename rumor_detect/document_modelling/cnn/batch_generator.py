# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import numpy as np
from sklearn.utils import shuffle

from parameter import *


def decode_file_(path, PADDING_TOKEN):
    """
    Decoding a data file into numpy data.
    Input
    - path: data path.
    - PADDING_TOKEN: an integer represents a padding token.
    Output
    - X: two dimensional numpy array.
    - y: labels, one dimensional numpy array.
    """
    X_list, y_list = [], []
    with open(path) as f:
        for line in f:
            args = line.strip().split(",")
            x = args[1].split()
            if len(x) < SEQUENCE_LENGTH:
                remain = SEQUENCE_LENGTH - len(x)
                for i in range(remain):
                    x.append(PADDING_TOKEN)
            else:
                x = x[:SEQUENCE_LENGTH]
            X_list.append(x)
            y_list.append(int(args[0]))
    X, y = np.array(X_list, dtype=np.int32), np.array(y_list, dtype=np.int32)
    return X, y


class BatchGenerator(object):
    """
    An object for generating batch feeding into tensorflow.
    For processing convenience, sentence length should be fixed to a number
    FIXED_LENGTH. Short sentence  will padding a sequence of special symbol
    <PAD>, util its length equals to FIXED_LENGTH. Long sentence will discard
    its exceeding parts.
    """
    def __init__(self, path, vocab):
        self.X_, self.y_ = decode_file_(path, vocab.padding_token_idx())
        self.iter_in_epoch_ = 0

    def next_batch(self):
        """
        Fetching next batch of data.
        Output
        - next_X: BATCH_NUM_ x FIXED_LENGTH_, two dimensional array.
        - next_y: BATCH_NUM, one dimensional array.
        """
        start_idx = self.iter_in_epoch_ * BATCH_NUM
        end_idx = start_idx + BATCH_NUM
        next_X = self.X_[start_idx:end_idx]
        next_y = self.y_[start_idx:end_idx]

        # shuffle per epoch
        self.iter_in_epoch_ += 1
        total_iter_in_epoch = self.X_.shape[0] / BATCH_NUM
        if self.iter_in_epoch_ >= total_iter_in_epoch:
            self.X_, self.y_ = shuffle(self.X_, self.y_)
            self.iter_in_epoch_ = 0

        return next_X, next_y

    def num_iter_per_epoch(self):
        return int(len(self.X_) / BATCH_NUM)
