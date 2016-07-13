# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, \
    recall_score, precision_score


def print_metrics(y_valid, y_predict):
    accuracy = accuracy_score(y_valid, y_predict)
    precision = precision_score(y_valid, y_predict)
    recall = recall_score(y_valid, y_predict)
    f1 = f1_score(y_valid, y_predict)
    print 'accuracy: %.4f, precision: %.4f, recall: %.4f, f1: %.4f' \
          % (accuracy, precision, recall, f1)
