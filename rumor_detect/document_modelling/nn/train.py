# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import logging
import json
from argparse import ArgumentParser

from rumor_detect.document_modelling.nn.common.rumor_corpus_for_nn import \
    RumorCorpusForNN
from rumor_detect.document_modelling.nn.common.word2vec_lookup_table import \
    Word2VecLookupTable
from rumor_detect.document_modelling.nn.rnn.lstm_avg.classifier import \
    LSTMAverageClassifier
from rumor_detect.document_modelling.nn.rnn.lstm_standard.classifier import \
    LSTMStandardClassifier
from rumor_detect.document_modelling.nn.softmax.classifier import \
    SoftmaxClassifier
from rumor_detect.document_modelling.nn.cnn.classifier import CNNClassifier

logger = logging.getLogger("nn.train")


def construct_classifier(config):
    constructor = {
        'SOFTMAX': SoftmaxClassifier,
        'CNN': CNNClassifier,
        'LSTM_STANDARD': LSTMStandardClassifier,
        'LSTM_AVG': LSTMAverageClassifier
    }[config['CLASSIFER_TYPE']]
    clf = constructor(config)
    logger.info('get classifier %s' % type(clf).__name__)
    return clf


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s]: %(message)s",
        level=logging.INFO
    )

    parser = ArgumentParser()
    parser.add_argument(
        "train_path",
        help="Training data location."
    )
    parser.add_argument(
        "dev_path",
        help="validation data location."
    )
    parser.add_argument(
        "word2vec_path",
        help="model file for word2vec location."
    )
    parser.add_argument(
        "config_path",
        help="configuration json file location"
    )
    args = parser.parse_args()

    logger.info('loading configure file...')
    config = json.load(open(args.config_path))

    logger.info('building vocabulary from word2vec model file...')
    word2vec_lookup_table = Word2VecLookupTable(args.word2vec_path)

    logger.info('loading train corpus...')
    train_corpus = RumorCorpusForNN(
        path=args.train_path,
        word2vec_lookup_table=word2vec_lookup_table,
        fixed_size=config['SEQUENCE_MAX_LEN']
    )
    logger.info('loading validation corpus...')
    dev_corpus = RumorCorpusForNN(
        path=args.dev_path,
        word2vec_lookup_table=word2vec_lookup_table,
        fixed_size=config['SEQUENCE_MAX_LEN']
    )

    clf = construct_classifier(config)
    clf.run_train(train_corpus, dev_corpus, word2vec_lookup_table)
    clf.run_eval(dev_corpus, word2vec_lookup_table)

