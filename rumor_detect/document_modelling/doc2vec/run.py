# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import argparse
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
    def __init__(self, pos_path, neg_path):
        """
        Construct corpus iterator from files (pos_path and neg_path).
        Input:
        - pos_path: csv file containing positive documents.
        - neg_path: csv file containing negative documents.
        """
        self.pos_corpus_ = TextSegmentedCorpus(pos_path)
        self.neg_corpus_ = TextSegmentedCorpus(neg_path)

    def __iter__(self):
        """
        Iterate over positive documents and negative documents sequentially.
        Output:
        - tagged_doc: an doc2vec.TaggedDocument object. Positive documents are
        tagged as 'POS_<iter_num>'. Negative documents are tagged as
        'NEG_<iter_num>'.
        """
        for i, sentence in enumerate(self.pos_corpus_):
            yield TaggedDocument(sentence, ["POS_%d" % i])
            self.pos_count_ = i
        for i, sentence in enumerate(self.neg_corpus_):
            yield TaggedDocument(sentence, ["NEG_%d" % i])
            self.neg_count_ = i

    def pos_count(self):
        return self.pos_count_

    def neg_count(self):
        return self.neg_count_


def grid_search(pos_path, neg_path):
    logger = logging.getLogger("doc2vec.run")

    logger.info("Training Doc2Vec")
    corpus = HelperCorpus(pos_path, neg_path)
    model = Doc2Vec(corpus, size=100, alpha=.025, window=8, min_count=5,
                    sample=0, hs=1, dm_concat=1, workers=4)

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
        "pos_path",
        help="positive instances file path used for validation"
    )
    parser.add_argument(
        "neg_path",
        help="negative instances file path used for validation"
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    grid_search(args.pos_path, args.neg_path)