# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import codecs
import numpy as np

UNKNOWN_TOKEN_ = "<UNK>"
PADDING_TOKEN_ = "<PAD>"


class Word2VecLookupTable:
    def __init__(self, path):
        """ Construct lookup table from word2vec model file.

        Input
        - path: word2vec model file path.

        Variables:
        - num_words: number of words in word2vec file.
        - dimension_size: dimension of embedding in word2vec model.
        - word2idx: a mapping for word to index.
        - idx2word: a mapping for index to word.
        """
        self.path_ = path
        with codecs.open(path, 'r', 'utf-8') as in_f:
            header = in_f.readline().strip()
            args = header.split()
            self.dimension_size_ = int(args[1])

            # construct word to index, index to word
            word2idx, idx2word = {}, {}
            for i, line in enumerate(in_f):
                word = line.strip().split()[0]
                word2idx[word] = i
                idx2word[i] = word

            # add special symbol for padding token
            word2idx[UNKNOWN_TOKEN_] = len(word2idx)
            word2idx[PADDING_TOKEN_] = len(word2idx)
            idx2word[len(idx2word)] = UNKNOWN_TOKEN_
            idx2word[len(idx2word)] = PADDING_TOKEN_

            self.num_words_ = len(word2idx)
            self.word2idx_ = word2idx
            self.idx2word_ = idx2word

    def embedding_matrix(self):
        W = np.random.uniform(
            low=-1, high=1,
            size=[self.num_words_, self.dimension_size_]
        ).astype(np.float32)
        with codecs.open(self.path_, 'r', 'utf-8') as in_f:
            in_f.readline()  # skip header
            for i, line in enumerate(in_f):
                W[i, :] = [float(num) for num in line.strip().split()[1:]]
        return W

    def get_word_idx(self, word):
        if word in self.word2idx_:
            return self.word2idx_[word]
        else:
            return self.word2idx_[UNKNOWN_TOKEN_]

    def get_padding_word_idx(self):
        return self.word2idx_[PADDING_TOKEN_]
