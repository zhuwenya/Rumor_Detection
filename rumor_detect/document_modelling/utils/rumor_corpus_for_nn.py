# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import codecs
import numpy as np
import sklearn.utils


def truncate_or_pad_(sequence, padding_idx, max_size):
    origin_length = len(sequence)
    if origin_length >= max_size:
        fixed_length_seq = sequence[:max_size]
        seq_length = max_size
    else:
        remain_length = max_size - origin_length
        fixed_length_seq = sequence + [padding_idx] * remain_length
        seq_length = origin_length
    return fixed_length_seq, seq_length


class RumorCorpusForNN:
    def __init__(self, path, word2vec_lookup_table, fixed_size):
        sequences, labels, lengths = [], [], []
        with codecs.open(path, 'r', 'utf-8') as in_f:
            for line in in_f:
                args = line.strip().split()
                sequence = [word2vec_lookup_table.get_word_idx(word)
                            for word in args[1:]]
                sequence, length = truncate_or_pad_(
                    sequence,
                    word2vec_lookup_table.get_padding_word_idx(),
                    fixed_size
                )
                if length != 0:
                    sequences.append(sequence)
                    labels.append(int(args[0]))
                    lengths.append(length)

        self.sequences_ = np.array(sequences, dtype=np.int32)
        self.labels_ = np.array(labels, dtype=np.int32)
        self.lengths_ = np.array(lengths, dtype=np.int32)
        self.shuffle_()

        self.iter_num_ = 0
        self.num_instances_ = self.sequences_.shape[0]

    def next_batch(self, batch_size):
        assert self.num_instances_ >= batch_size

        if self.iter_num_ + batch_size > self.num_instances_:
            self.shuffle_()
            self.iter_num_ = 0

        start_idx = self.iter_num_
        end_idx = self.iter_num_ + batch_size
        X = self.sequences_[start_idx:end_idx]
        y = self.labels_[start_idx:end_idx]
        lengths = self.lengths_[start_idx:end_idx]

        self.iter_num_ = end_idx

        return X, y, lengths

    def shuffle_(self):
        self.sequences_, self.labels_, self.lengths_ = sklearn.utils.shuffle(
            self.sequences_,
            self.labels_,
            self.lengths_
        )

    def num_iter_in_epoch(self, batch_size):
        return int(self.num_instances_ / batch_size)
