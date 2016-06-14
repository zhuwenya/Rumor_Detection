# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import argparse
import codecs
import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle


def construct_dataset(pos_file, neg_file):
    """
    Construct data set from pos_file and neg_file object.
    Each line in those files is a segmented document.

    Input:
    - pos_file: file object containing positive instances.
    - neg_file: file object containing negative instances.
    Output:
    - X: list of documents.
    - y: document labels.
    """
    X_pos = [line.strip() for line in pos_file]
    X_neg = [line.strip() for line in neg_file]
    y_pos = np.ones(len(X_pos), dtype=np.int32)
    y_neg = -1 * np.ones(len(X_neg), dtype=np.int32)
    X = X_pos + X_neg
    y = np.concatenate((y_pos, y_neg))
    X, y = shuffle(X, y, random_state=12345)
    return X, y


def tfidf_param_search(X, y):
    """
    Cross validation search good parameter for TFIDF.

    Input:
    -X: list of strings, each string represents a segmented doc.
    -y: labels for X.
    """
    print "Cross validation for tfidf"
    param_grid=[
        {'tfidf__lowercase': [True, False]},
        {'tfidf__max_df': np.linspace(0.2, 1.0, 9)},
        {'tfidf__min_df': [5, 10, 20, 50, 100]},
        {'tfidf__max_features': [10000, 50000, 100000, 500000]},
        {'tfidf__norm': ['l1', 'l2', None]},
        {'tfidf__use_idf': [True, False]}
    ]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), dtype=np.int32)
    lr = LogisticRegression()
    pipeline = Pipeline([('tfidf', tfidf), ('lr', lr)])
    cv = GridSearchCV(pipeline, param_grid, refit=False, n_jobs=3, verbose=2)
    cv.fit(X, y)

    for t in cv.grid_scores_:
        print t.parameters, t.mean_validation_score


def lr_param_search(X, y):
    """
    Cross validation search good parameter for LogisticRegression.

    Input:
    -X: list of strings, each string represents a segmented doc.
    -y: labels for X.
    """
    print "Cross validation for lr"
    param_grid = [
        {'lr__C': [100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5]},
    ]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), dtype=np.int32, lowercase=True,
                            max_df=0.5, min_df=5, max_features=500000,
                            norm='l2', use_idf=True)
    lr = LogisticRegression()
    pipeline = Pipeline([('tfidf', tfidf), ('lr', lr)])
    cv = GridSearchCV(pipeline, param_grid, refit=False, n_jobs=3, verbose=2)
    cv.fit(X, y)

    for t in cv.grid_scores_:
        print t.parameters, t.mean_validation_score


if __name__ == "__main__":
    """
    Running parameter search cross validation for small data set.
    Large data set experiment haven't been test.
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
    args = parser.parse_args()
    with codecs.open(args.pos_path, "r", "utf-8") as pos_file,\
        codecs.open(args.neg_path, "r", "utf-8") as neg_file:
        X, y = construct_dataset(pos_file, neg_file)

    # Searching for good tfidf param
    # tfidf_param_search(X, y)

    # Search for good LR param
    # lr_param_search(X, y)

    # Best parameter
    tfidf = TfidfVectorizer(ngram_range=(1, 2), dtype=np.int32, lowercase=True,
                            max_df=0.5, min_df=5, max_features=500000,
                            norm='l2', use_idf=True)
    lr = LogisticRegression(C=10)
    pipeline = Pipeline([('tfidf', tfidf), ('lr', lr)])
    score = cross_val_score(pipeline, X, y, n_jobs=3)
    print "Best parameter cv score", score
