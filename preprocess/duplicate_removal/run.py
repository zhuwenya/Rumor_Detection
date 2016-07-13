# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import codecs
from sklearn.feature_extraction.text import TfidfVectorizer


class Corpus(object):
    def __init__(self, path):
        self.path_ = path

    def __iter__(self):
        with codecs.open(self.path_, 'r', 'utf-8') as in_file:
            for line in in_file:
                yield line.strip()


def duplicate_remove(sentences, transformer, threshold=0.8):
    X = transformer.fit_transform(sentences)
    mask = []
    scores = X.dot(X.T)
    for i, x in enumerate(X):
        if i == 0:
            next_mask = True
        else:
            next_mask = not (scores[i, :i].todense() >= threshold).any()
        mask.append(next_mask)

    ret = [sentence for i, sentence in enumerate(sentences) if mask[i]]
    return ret


def reference_duplicate_remove(sentences, ref_sentences,
                               transformer, threshold=0.8):
    all_sentences = sentences + ref_sentences
    X = transformer.fit_transform(all_sentences)
    X_src = X[:len(sentences)]
    X_dst = X[len(sentences):]
    scores = X_src.dot(X_dst.T)
    mask = []
    for i in range(len(sentences)):
        next_mask = not (scores[i, :].todense() >= threshold).any()
        mask.append(next_mask)
    return [sentence for i, sentence in enumerate(sentences) if mask[i]]


if __name__ == "__main__":
    corpus = Corpus('../../dataset/label/rumor_valid.csv')
    ref_corpus = Corpus('../../dataset/label/rumor_train.csv')
    sentences = [line for line in corpus]
    ref_sentences = [line for line in ref_corpus]
    print 'original rumor size:', len(sentences)

    transformer = TfidfVectorizer(input=sentences, min_df=5,
                                  max_df=0.5, max_features=100000)
    sentences_2 = duplicate_remove(sentences, transformer)
    print 'after remove duplicated sentences, size:', len(sentences_2)

    sentences_3 = reference_duplicate_remove(
        sentences_2, ref_sentences, transformer
    )
    print 'after remove appeared sentences, size:', len(sentences_3)

    with codecs.open('../../dataset/label_duplicate_removal/rumor_valid.data',
                     'w', 'utf-8') as out_file:
        for sentence in sentences_3:
            out_file.write(sentence + '\n')
