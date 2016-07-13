# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import codecs
import numpy as np
import logging
import sklearn.utils
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from rumor_detect.document_modelling.utils.metrics import print_metrics

TRAIN_PATH = '../data/train.data'
VALID_PATH = '../data/valid.data'


class HelperCorpus:
    def __init__(self):
        train_pos, train_neg = 0, 0
        with codecs.open(TRAIN_PATH, 'r', 'utf-8') as in_file:
            for line in in_file:
                args = line.strip().split()
                label = int(args[0])
                train_pos += (label == 1)
                train_neg += (label == 0)

        valid_pos, valid_neg = 0, 0
        with codecs.open(VALID_PATH, 'r', 'utf-8') as in_file:
            for line in in_file:
                args = line.strip().split()
                label = int(args[0])
                valid_pos += (label == 1)
                valid_neg += (label == 0)

        self.train_pos_ = train_pos
        self.train_neg_ = train_neg
        self.valid_pos_ = valid_pos
        self.valid_neg_ = valid_neg

    def __iter__(self):
        train_pos_i, train_neg_i = 0, 0
        with codecs.open(TRAIN_PATH, 'r', 'utf-8') as in_file:
            for line in in_file:
                args = line.strip().split()
                label = int(args[0])
                sentence = args[1:]
                if label == 1:
                    tag = 'TRAIN_POS_%d' % train_pos_i
                    train_pos_i += 1
                else:
                    tag = 'TRAIN_NEG_%d' % train_neg_i
                    train_neg_i += 1
                yield TaggedDocument(sentence, [tag])

        valid_pos_i, valid_neg_i = 0, 0
        with codecs.open(VALID_PATH, 'r', 'utf-8') as in_file:
            for line in in_file:
                args = line.strip().split()
                label = int(args[0])
                sentence = args[1:]
                if label == 1:
                    tag = 'VALID_POS_%d' % valid_pos_i
                    valid_pos_i += 1
                else:
                    tag = 'VALID_NEG_%d' % valid_neg_i
                    valid_neg_i += 1
                yield TaggedDocument(sentence, [tag])

                
def run(dim=400):
    logger = logging.getLogger("doc2vec.run")

    logger.info('calculating doc2vec model...')
    corpus = HelperCorpus()
    model_dm = Doc2Vec(corpus, size=dim, alpha=0.025, window=4,
                       min_count=5, sample=1e-5, hs=0, negative=10,
                       dm=1, dm_concat=0, dbow_words=0, iter=15, workers=7)
    model_dbow = Doc2Vec(corpus, size=dim, alpha=0.025, window=4,
                         min_count=5, sample=1e-5, hs=0, negative=10,
                         dm=0, dm_concat=0, dbow_words=0, iter=15, workers=7)

    logger.info('retrieving training vectors from model...')
    X_train = np.zeros((corpus.train_pos_ + corpus.train_neg_, dim * 2))
    y_train = np.zeros(corpus.train_pos_ + corpus.train_neg_, dtype=np.int32)
    for pos_i in range(corpus.train_pos_):
        X_train[pos_i, :dim] = model_dm.docvecs['TRAIN_POS_%d' % pos_i]
        X_train[pos_i, dim:] = model_dbow.docvecs['TRAIN_POS_%d' % pos_i]
        y_train[pos_i] = 1
    for neg_i in range(corpus.train_neg_):
        X_train[corpus.train_pos_ + neg_i, :dim] = \
            model_dm.docvecs['TRAIN_NEG_%d' % neg_i]
        X_train[corpus.train_pos_ + neg_i, dim:] = \
            model_dbow.docvecs['TRAIN_NEG_%d' % neg_i]
        y_train[corpus.train_pos_ + neg_i] = 0
    X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

    logger.info('training logistic regression')
    lr = LogisticRegression(C=10)
    lr.fit(X_train, y_train)

    logger.info('retrieving validation vectors from model...')
    X_valid = np.zeros((corpus.valid_pos_ + corpus.valid_neg_, dim * 2))
    y_valid = np.zeros(corpus.valid_pos_ + corpus.valid_neg_, dtype=np.int32)
    for pos_i in range(corpus.valid_pos_):
        X_valid[pos_i, :dim] = model_dm.docvecs['VALID_POS_%d' % pos_i]
        X_valid[pos_i, dim:] = model_dbow.docvecs['VALID_POS_%d' % pos_i]
        y_valid[pos_i] = 1
    for neg_i in range(corpus.valid_neg_):
        X_valid[corpus.valid_pos_ + neg_i, :dim] = \
            model_dm.docvecs['VALID_NEG_%d' % neg_i]
        X_valid[corpus.valid_pos_ + neg_i, dim:] = \
            model_dbow.docvecs['VALID_NEG_%d' % neg_i]
        y_valid[corpus.valid_pos_ + neg_i] = 0

    logger.info('calculating results...')
    y_valid_predict = lr.predict(X_valid)
    print_metrics(y_valid, y_valid_predict)

if __name__ == "__main__":
    """
    Running parameter search cross validation for small data set.
    """
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    run()
