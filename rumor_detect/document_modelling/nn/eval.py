# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
from argparse import ArgumentParser
import json
import logging
from rumor_detect.document_modelling.nn.common.rumor_corpus_for_nn import \
    RumorCorpusForNN
from rumor_detect.document_modelling.nn.common.word2vec_lookup_table import \
    Word2VecLookupTable
from rumor_detect.document_modelling.nn.train import construct_classifier

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s]: %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger('nn.eval')

    parser = ArgumentParser()
    parser.add_argument(
        "valid_path",
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

    logger.info('loading validation corpus...')
    valid_corpus = RumorCorpusForNN(
        path=args.valid_path,
        word2vec_lookup_table=word2vec_lookup_table,
        fixed_size=config['SEQUENCE_MAX_LEN']
    )

    clf = construct_classifier(config)
    clf.run_eval(valid_corpus, word2vec_lookup_table)

