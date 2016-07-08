# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import argparse

import logging
from gensim.models.word2vec import Word2Vec, LineSentence


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s]: %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger('word2vec.run')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "doc_path",
        help="path to document file"
    )
    parser.add_argument(
        "save_path",
        help="word2vec model save path"
    )
    args = parser.parse_args()

    sentences = LineSentence(args.doc_path)
    model = Word2Vec(sentences, size=100, min_count=30, workers=7, negative=10,
                     iter=10, sg=1)
    model.save_word2vec_format(args.save_path)
