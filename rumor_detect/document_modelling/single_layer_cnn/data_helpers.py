# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
from collections import defaultdict

PADDING_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"

def build_vocab(readers, min_count=5):
    """
    Build vocabulary from the content of readers.
    Input
    - readers: instances of type rumor.detect.document_modelling.preprocessing.corpus.TextSegmentedCorpus.
    - min_count: words appear less than min_count will be converted to a special symbol <UNK>.
    Output
    - word_to_idx: an map object that map word to numerical index.
    """
    word_count_table = defaultdict(int)

    for reader in readers:
        for sentence in reader:
            for word in sentence:
                word_count_table[word] += 1

    word_to_idx = {
        PADDING_TOKEN: 0,
        UNKNOWN_TOKEN: 1
    }

    for k, v in word_count_table.iteritems():
        if v >= min_count:
            count_so_far = len(word_to_idx)
            word_to_idx[k] = count_so_far


    return word_to_idx


def encode_sentence(sentence, word_to_idx):
    """
    Encoding sentence according to code bode word_to_idx.
    Input
    - sentence: a list of string, each string is a word.
    - word_to_idx: an map object that map word to numerical index.
    Output
    - encode_str: a string, sentence encoded by word_to_idx code book.
    """
    idx_str_list = []
    for word in sentence:
        word = word if word in word_to_idx else UNKNOWN_TOKEN
        idx_str = str(word_to_idx[word])
        idx_str_list.append(idx_str)

    encode_str = ' '.join(idx_str_list)
    return encode_str


def save_word_to_idx(path, word_to_idx):
    """
    Save word_to_idx map object to file.
    Input
    - path: path to save.
    - word_to_idx: an map object that map word to numerical index.
    """
    with open(path, "w", "utf-8") as f:
        for k, v in word_to_idx:
            f.write("%s %d\n" % (k, v))


def load_word_to_idx(path):
    """
    Load word_to_idx map object from a file.
    Input
    - path: path to load.
    Output
    - word_to_idx: an map object that map word to numerical index.
    """
    word_to_idx = {}
    with open(path, "r", "utf-8") as f:
        for line in f:
            args = line.strip().split()
            word, idx = args[0], int(args[1])
            word_to_idx[word] = idx
    return word_to_idx