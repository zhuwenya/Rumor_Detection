# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import codecs
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from rumor_detect.document_modelling.utils.metrics import print_all


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
    print 'reading data...'
    X_train, y_train = read_data('../data/train_1_2.data')
    X_valid, y_valid = read_data('../data/valid_1_100.data')

    # Searching for good tfidf param
    # tfidf_param_search(X_train, y_train)

    # Search for good LR param
    # lr_param_search(X_train, y_train)

    print 'fitting parameters...'
    tfidf = TfidfVectorizer(ngram_range=(1, 2), dtype=np.int32, lowercase=True,
                            max_df=0.5, min_df=5, max_features=500000,
                            norm='l2', use_idf=True)
    lr = LogisticRegression(C=10, class_weight='balanced')
    pipeline = Pipeline([('tfidf', tfidf), ('lr', lr)])
    pipeline.fit(X_train, y_train)

    # print metrics
    idx = np.where(pipeline.classes_ == 1)[0][0]
    y_score = pipeline.predict_proba(X_valid)[:, idx]
    print_all(y_valid, y_score)
