# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import codecs

import os
from argparse import ArgumentParser
from sklearn.utils import shuffle

from rumor_detect.document_modelling.preprocessing.corpus import TextCorpus

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
        "data_dir",
        help="directory to save data"
    )
    args = parser.parse_args()
    pos_reader = TextCorpus(args.rumor_csv)
    neg_reader = TextCorpus(args.doc_csv)

    # encoded sentences according to vocabulary
    pos_sentences = [sentence for sentence in pos_reader]
    neg_sentences = [sentence for sentence in neg_reader]
    labels = [1 for i in range(len(pos_sentences))] + \
             [0 for i in range(len(neg_sentences))]
    sentences = pos_sentences + neg_sentences

    # shuffle data
    sentences, labels = shuffle(sentences, labels)

    # split data, save into directory.
    SPLIT_PROTION = 0.8
    total_num = len(sentences)
    train_num = int(total_num * SPLIT_PROTION)
    train_path = os.path.join(args.data_dir, "train.data")
    dev_path = os.path.join(args.data_dir, "dev.data")

    with codecs.open(train_path, "w", "utf-8") as train_file, \
        codecs.open(dev_path, "w", "utf-8") as dev_fie:
        for i in range(train_num):
            train_file.write("%d %s\n" % (labels[i], sentences[i]))
        for i in range(train_num, total_num):
            dev_fie.write("%d %s\n" % (labels[i], sentences[i]))
