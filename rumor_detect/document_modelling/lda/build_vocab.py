# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import logging
import codecs

from gensim.corpora import Dictionary

logger = logging.getLogger('lda.build_vocab')


class UnlabeledCorpus(object):
    def __init__(self, path, stopwords=None):
        self.path_ = path
        self.stopwords_ = stopwords

    def __iter__(self):
        with codecs.open(self.path_, 'r', 'utf-8') as in_file:
            for line in in_file:
                doc = [word for word in line.strip().split() if len(word) > 0]
                if self.stopwords_ is not None:
                    doc = [word for word in doc if word not in self.stopwords_]
                yield doc


def load_stopwords(path):
    logger.info('Loading stop words from ' + path)
    stopwords = set()
    with codecs.open(path, 'r', 'utf-8') as f:
        for line in f:
            word = line.strip()
            if len(word) > 0: stopwords.add(word)
    return stopwords


def build_vocab(path, stopwords):
    logger.info('Building vocabulary from ' + path)
    corpus = UnlabeledCorpus(path, stopwords)
    vocab = Dictionary(corpus)
    vocab.filter_extremes()
    return vocab

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    stopwords = load_stopwords('./stopwords.txt')
    vocab = build_vocab('../data/all.data', stopwords)
    vocab.save_as_text('./vocab.txt', sort_by_word=False)