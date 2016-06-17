# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import argparse
import random
import numpy as np
import logging
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from rumor_detect.document_modelling.preprocessing.corpus \
    import TextSegmentedCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class HelperCorpus:
    """
    Corpus helper for training doc2vec.
    """
    def __init__(self, path, prefix):
        """
        Construct corpus iterator from files.
        Input:
        - path: csv file containing documents.
        """
        self.corpus_ = TextSegmentedCorpus(path)
        self.prefix_ = prefix

    def __iter__(self):
        """
        Iterate over documents sequentially.
        Output:
        - tagged_doc: an doc2vec.TaggedDocument object. Each document is tagged
         as <prefix>_<idxnum>
        """
        for i, sentence in enumerate(self.corpus_):
            yield TaggedDocument(sentence, ["%s_%d" % (self.prefix_, i)])
            self.count_ = i + 1

    def count(self):
        return self.count_


class MemoryHelperCorpus:
    def __init__(self, pos_path, neg_path):
        pos_corpus = HelperCorpus(pos_path, "POS")
        neg_corpus = HelperCorpus(neg_path, "NEG")
        self.sentences_ = [sentence for sentence in pos_corpus]
        self.sentences_.extend([sentence for sentence in neg_corpus])
        self.pos_count_ = pos_corpus.count()
        self.neg_count_ = neg_corpus.count()

    def __iter__(self):
        random.shuffle(self.sentences_)
        for sentence in self.sentences_:
            yield sentence

    def pos_count(self):
        return self.pos_count_

    def neg_count(self):
        return self.neg_count_


def grid_search(doc_path, pos_path, neg_path):
    logger = logging.getLogger("doc2vec.run")

    # pretrain doc from a large dataset
    doc_corpus = HelperCorpus(doc_path, "DOC")
    model = Doc2Vec(doc_corpus, size=100, alpha=0.025, window=4, min_count=5,
                    sample=0, hs=0, negative=5, dm_concat=1, iter=5, workers=7)


    # fix random seed to remove perturbation
    logger.info("Feeding corpus into memory")
    corpus = MemoryHelperCorpus(pos_path, neg_path)
    logger.info("Training Doc2Vec")
    model.train(corpus)

    logger.info("Retrieving Doc2Vec")
    X = np.zeros((corpus.pos_count() + corpus.neg_count(), 100))
    y = np.zeros(corpus.pos_count() + corpus.neg_count(), dtype=np.int32)
    for pos_i in range(corpus.pos_count()):
        X[pos_i] = model.docvecs['POS_%d' % pos_i]
        y[pos_i] = 1
    for neg_i in range(corpus.neg_count()):
        X[corpus.pos_count() + neg_i] = model.docvecs['NEG_%d' % neg_i]
        y[neg_i] = -1

    logger.info("Training LR")
    lr = LogisticRegression()
    scores = cross_val_score(lr, X, y)
    score = np.mean(scores)
    logger.info("Cross validation score for LR: %.3f" % score)


if __name__ == "__main__":
    """
    Running parameter search cross validation for small data set.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "doc_path",
        help="document instances file path used for training doc2vec"
    )
    parser.add_argument(
        "pos_path",
        help="positive instances file path "
    )
    parser.add_argument(
        "neg_path",
        help="negative instances file path"
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    grid_search(args.doc_path, args.pos_path, args.neg_path)