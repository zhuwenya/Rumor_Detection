# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import codecs
import pickle
from collections import defaultdict


PADDING_TOKEN_ = "<PAD>"
UNKNOWN_TOKEN_ = "<UNK>"


class Vocabulary(object):
    def __init__(self, lower_case=True):
        self.lower_case_ = lower_case
        self.word_to_idx_ = {
            PADDING_TOKEN_: 0,
            UNKNOWN_TOKEN_: 1
        }

    def build(self, readers, min_df=5):
        """
        Build vocabulary from the content of readers.
        Input
        - readers: instances of type preprocessing.corpus.TextSegmentedCorpus.
        - min_df: word document frequency less than min_df will be discard.
        """

        # build word to document frequency table.
        df_count_table = defaultdict(int)
        for reader in readers:
            for sentence in reader:
                word_set = {word.lower() if self.lower_case_ else word for word in sentence}
                for word in word_set:
                    df_count_table[word] += 1

        # build word to index table.
        for k, v in df_count_table.iteritems():
            if v >= min_df:
                count_so_far = len(self.word_to_idx_)
                self.word_to_idx_[k] = count_so_far

    def save(self, path):
        """
        Save object to file.
        Input
        - path: path to save.
        """
        with codecs.open(path, "w", "utf-8") as f:
            f.write("lower_case %d\n" % self.lower_case_)
            for k, v in self.word_to_idx_.iteritems():
                f.write("%s %d\n" % (k, v))

    @staticmethod
    def load(path):
        """
        Load object from a file.
        Input
        - path: path to load.
        Output
        - vocab: an vocabulary object.
        """
        vocab = None
        with codecs.open(path, "r", "utf-8") as f:
            header = f.readline().strip()
            lower_case = bool(header.split()[1])
            word_to_idx = {}
            for line in f:
                args = line.strip().split()
                k, v = args[0], int(args[1])
                word_to_idx[k] = v
            vocab = Vocabulary(lower_case=lower_case)
            vocab.word_to_idx_ = word_to_idx
        return vocab

    def encode_sentence(self, sentence):
        """
        Encoding sentence according to code bode word_to_idx.
        Input
        - sentence: a list of string, each string is a word.
        Output
        - encode_str: a string, sentence encoded by word_to_idx code book.
        """
        idx_str_list = []
        for word in sentence:
            idx = self.word_idx(word)
            idx_str = str(idx)
            idx_str_list.append(idx_str)

        encode_str = ' '.join(idx_str_list)
        return encode_str

    def encode_sentences(self, reader):
        """
        Encoding all sentences in reader according to code book word_to_idx.
        Input:
        - reader: instance of type preprocessing.corpus.TextSegmentedCorpus.
        Output:
        - encode_strs: a list of string, each string is a sentence encoded
          by word_to_idx.
        """
        encode_strs = [self.encode_sentence(sentence) for sentence in reader]
        return encode_strs

    def number_words(self):
        """
        Get number of words in the vocabulary.
        Output:
        - num_words: number of words in the vocabulary.
        """
        num_words = len(self.word_to_idx_)
        return num_words

    def word_idx(self, word):
        word = word.lower() if self.lower_case_ else word
        word = word if word in self.word_to_idx_ else UNKNOWN_TOKEN_
        return self.word_to_idx_[word]

    def padding_token_idx(self):
        """
        Get padding token index.
        Output
        - idx: padding token index in vocabulary.
        """
        idx = self.word_to_idx_.get(PADDING_TOKEN_)
        return idx