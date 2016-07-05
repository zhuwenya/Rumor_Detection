# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import argparse
import codecs
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def read_data(path):
    X, y = [], []
    with codecs.open(path, 'r', 'utf-8') as in_file:
        for line in in_file:
            args = line.strip().split()
            label = int(args[0])
            x = ' '.join(args[1:])
            X.append(x)
            y.append(label)
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
        "train_path",
        help="train file path"
    )
    parser.add_argument(
        "valid_path",
        help="validation file path"
    )
    args = parser.parse_args()
    X_train, y_train = read_data(args.train_path)
    X_valid, y_valid = read_data(args.valid_path)

    # Searching for good tfidf param
    # tfidf_param_search(X_train, y_train)

    # Search for good LR param
    # lr_param_search(X_train, y_train)

    # Best parameter
    tfidf = TfidfVectorizer(ngram_range=(1, 2), dtype=np.int32, lowercase=True,
                            max_df=0.5, min_df=5, max_features=500000,
                            norm='l2', use_idf=True)
    lr = LogisticRegression(C=10)
    pipeline = Pipeline([('tfidf', tfidf), ('lr', lr)])
    pipeline.fit(X_train, y_train)
    y_valid_predict = pipeline.predict(X_valid)
    print classification_report(y_valid, y_valid_predict)
