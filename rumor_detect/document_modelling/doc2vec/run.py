# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import argparse
import random
import numpy as np
import logging
import sklearn.utils
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from rumor_detect.document_modelling.utils.corpus \
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

    
class CombinedCorpus:
    def __init__(self):
        self.corpus_list_ = []

    def add(self, corpus):
        self.corpus_list_.append(corpus)

    def __iter__(self):
        for corpus in self.corpus_list_:
            for sentence in corpus:
                yield sentence

                
class MemoryShuffleCorpus:
    def __init__(self, base_corpus):
        sentences = []
        for sentence in base_corpus:
            sentences.append(sentence)
        self.sentences_ = sentences

    def __iter__(self):
        random.shuffle(self.sentences_)
        for sentence in self.sentences_:
            yield sentence

            
def grid_search(pos_path, neg_path):
    logger = logging.getLogger("doc2vec.run")

    # pretrain doc from a large dataset
    pos_corpus = HelperCorpus(pos_path, "POS")
    neg_corpus = HelperCorpus(neg_path, "NEG")
    combined_corpus = CombinedCorpus()
    combined_corpus.add(pos_corpus)
    combined_corpus.add(neg_corpus)
    corpus = MemoryShuffleCorpus(combined_corpus)
    # model_dm = Doc2Vec(corpus, size=400, alpha=0.025, window=4,
    #                    min_count=5, sample=1e-5, hs=0, negative=15,
    #                    dm=1, dm_concat=0, dbow_words=0, iter=20, workers=7)
    model_dbow = Doc2Vec(corpus, size=400, alpha=0.025, window=4,
                         min_count=5, sample=1e-5, hs=0, negative=15,
                         dm=0, dm_concat=0, dbow_words=0, iter=20, workers=7)

    X = np.zeros((pos_corpus.count() + neg_corpus.count(), 400))
    y = np.zeros(pos_corpus.count() + neg_corpus.count(), dtype=np.int32)
    for pos_i in range(pos_corpus.count()):
        X[pos_i] = model_dbow.docvecs['POS_%d' % pos_i]
        y[pos_i] = 1
    for neg_i in range(neg_corpus.count()):
        X[pos_corpus.count() + neg_i] = model_dbow.docvecs['NEG_%d' % neg_i]
        y[pos_corpus.count() + neg_i] = -1
    X, y = sklearn.utils.shuffle(X, y)

    for C in [100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5]:
        logger.info("Training LR")
        lr = LogisticRegression(C=C)
        scores = cross_val_score(lr, X, y)
        score = np.mean(scores)
        logger.info("Cross validation score for LR (C=%f): %.3f" % (C, score))


if __name__ == "__main__":
    """
    Running parameter search cross validation for small data set.
    """

    parser = argparse.ArgumentParser()
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
    grid_search(args.pos_path, args.neg_path)
