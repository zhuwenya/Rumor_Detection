# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import os
from argparse import ArgumentParser

from sklearn.utils import shuffle

from rumor_detect.document_modelling.preprocessing.corpus import TextSegmentedCorpus
from rumor_detect.document_modelling.single_layer_cnn.preprocess.vocabulary import Vocabulary

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "rumor_csv",
        help="rumor.csv file location"
    )
    parser.add_argument(
        "doc_csv",
        help="document.csv file location"
    )
    parser.add_argument(
        "vocab",
        help="vocabulary file location"
    )
    parser.add_argument(
        "data_dir",
        help="directory to save data"
    )
    args = parser.parse_args()
    pos_reader = TextSegmentedCorpus(args.rumor_csv)
    neg_reader = TextSegmentedCorpus(args.doc_csv)

    # encoded sentences according to vocabulary
    vocab = Vocabulary.load(args.vocab)
    pos_sentences = vocab.encode_sentences(pos_reader)
    neg_sentences = vocab.encode_sentences(neg_reader)
    labels = [1 for i in range(len(pos_sentences))] + \
             [0 for i in range(len(neg_sentences))]
    encoded_senteces = pos_sentences + neg_sentences

    # shuffle data
    encoded_senteces, labels = shuffle(encoded_senteces, labels)

    # split data, save into directory.
    SPLIT_PROTION = 0.8
    total_num = len(encoded_senteces)
    train_num = int(total_num * SPLIT_PROTION)
    train_path = os.path.join(args.data_dir, "train.data")
    dev_path = os.path.join(args.data_dir, "dev.data")

    with open(train_path, "w") as train_file, \
        open(dev_path, "w") as dev_fie:
        for i in range(train_num):
            train_file.write("%d,%s\n" % (labels[i], encoded_senteces[i]))
        for i in range(train_num, total_num):
            dev_fie.write("%d,%s\n" % (labels[i], encoded_senteces[i]))





