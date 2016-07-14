# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
from sklearn.metrics import accuracy_score, f1_score, \
    recall_score, precision_score, precision_recall_curve, roc_auc_score
import numpy as np


def print_all(y_valid, y_score):
    y_pred = np.round(y_score)
    print_metrics(y_valid, y_pred)
    print_roc_auc(y_valid, y_score)
    print_under_target_recall(y_valid, y_score, target_recall=0.9)
    print_under_target_recall(y_valid, y_score, target_recall=0.95)


def print_metrics(y_valid, y_predict):
    accuracy = accuracy_score(y_valid, y_predict)
    precision = precision_score(y_valid, y_predict)
    recall = recall_score(y_valid, y_predict)
    f1 = f1_score(y_valid, y_predict)
    print 'accuracy: %.4f, precision: %.4f, recall: %.4f, f1: %.4f' \
          % (accuracy, precision, recall, f1)


def print_roc_auc(y_valid, y_score):
    print 'roc_auc: %f' % roc_auc_score(y_valid, y_score)


def print_under_target_recall(y_valid, y_score, target_recall=0.9):
    precision, recall, threshold = precision_recall_curve(y_valid, y_score)
    num_points = len(recall)
    for i in range(num_points)[::-1]:
        if recall[i] >= target_recall:
            msg = 'after adjust, accuracy: %.3f, precision: %.3f, ' \
                  'recall: %.3f, threshold: %.3f'
            accuracy = np.mean((y_score >= threshold[i]) == y_valid)
            value_tuple = (accuracy, precision[i], recall[i], threshold[i])
            print msg % value_tuple
            break
