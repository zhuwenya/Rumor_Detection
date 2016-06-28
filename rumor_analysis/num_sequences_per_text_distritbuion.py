# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>
import re
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

from rumor_detect.document_modelling.preprocessing.corpus import TextCorpus


PATTERN = u"，|。|！|？|；|……"


def get_num_sequences_per_text(corpus):
    num_sequences = []
    for sentence in corpus:
        next_num = len(re.split(PATTERN, sentence))
        num_sequences.append(next_num)
    return num_sequences


def get_num_words_per_sequence(corpus):
    num_words = []
    for i, sentence in enumerate(corpus):
        sub_sequences = re.split(PATTERN, sentence)
        for sub_sequence in sub_sequences:
            next = len([word for word in sub_sequence.split() if len(word) >= 0])
            num_words.append(next)
    return num_words


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "rumor_csv",
        help="Rumor text file location"
    )
    parser.add_argument(
        "doc_csv",
        help="Document text file location"
    )
    args = parser.parse_args()

    rumor_corpus = TextCorpus(args.rumor_csv)
    doc_corpus = TextCorpus(args.doc_csv)

    # num sequences per text
    rumor_num_sequences = get_num_sequences_per_text(rumor_corpus)
    doc_num_sequences = get_num_sequences_per_text(doc_corpus)
    print "mean number of sequences per text, (rumor: %f, doc: %f)" % \
          (np.mean(rumor_num_sequences), np.mean(doc_num_sequences))
    print "max number of sequences per text, (rumor: %f, doc: %f)" % \
          (np.max(rumor_num_sequences), np.max(doc_num_sequences))
    plt.style.use('ggplot')
    plt.hist([rumor_num_sequences, doc_num_sequences], 20, histtype='bar',
             cumulative=True, normed=1, label=['rumor', 'doc'])
    plt.legend()
    plt.xlabel("sequences per text")
    plt.ylabel("Cumulative Probability")
    plt.title("Number of Sequences In Rumor and Doc Text ")
    plt.show()

    # num words per sequences
    rumor_num_words = get_num_words_per_sequence(rumor_corpus)
    doc_num_words = get_num_words_per_sequence(doc_corpus)
    print "mean number of words per sequence, (rumor: %f, doc: %f)" % \
          (np.mean(rumor_num_words), np.mean(doc_num_words))
    print "max number of words per sequence, (rumor: %f, doc: %f)" % \
          (np.max(rumor_num_words), np.max(doc_num_words))
    plt.hist([rumor_num_words, doc_num_words], 100, histtype='bar',
             cumulative=True, range=(0, 1000), normed=1, label=['rumor', 'doc'])
    plt.legend()
    plt.xlabel("words per sequence")
    plt.ylabel("Cumulative Probability")
    plt.title("Number of Words In Sequence")
    plt.show()
