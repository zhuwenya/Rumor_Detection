# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from rumor_detect.document_modelling.ngrams.run import construct_dataset

if __name__ == "__main__":
    """
    Find top k words with largest weights.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pos_path",
        help="positive instances file path"
    )
    parser.add_argument(
        "neg_path",
        help="negative instances file path"
    )
    parser.add_argument(
        "k",
        help="words number to print"
    )
    args = parser.parse_args()
    sentences, y = construct_dataset(args.pos_path, args.neg_path)

    # build vocabulary and train LR
    tfidf = TfidfVectorizer(ngram_range=(1, 2), dtype=np.int32, lowercase=True,
                            max_df=0.5, min_df=5, max_features=500000,
                            norm='l2', use_idf=True)
    X = tfidf.fit_transform(sentences)
    lr = LogisticRegression(C=10)
    lr.fit(X, y)

    # Print top k words with largest weights.
    # Because word weight are affected by its inverse document frequency,
    # here we define $ weights\[word\] = lr.coef\[word\] * idf\[word\] $.
    k = int(args.k)
    names = tfidf.get_feature_names()
    weights = lr.coef_.flatten()

    if tfidf.idf_ is not None:
        weights *= tfidf.idf_

    sort_idx = np.argsort(weights)[::-1]
    for i, idx in enumerate(sort_idx[:k]):
        print "(%s, %f)" % (names[idx], weights[idx]),