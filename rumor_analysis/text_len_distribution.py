# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from rumor_detect.document_modelling.preprocessing.corpus \
    import TextSegmentedCorpus

if __name__ == "__main__":
    """
    Analzing the text length distribution of rumors.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "rumor_path",
        help="rumor csv file path"
    )
    parser.add_argument(
        "doc_path",
        help="doc csv file path"
    )

    args = parser.parse_args()
    rumor_corpus = TextSegmentedCorpus(args.rumor_path)
    doc_corpus = TextSegmentedCorpus(args.doc_path)

    # analyse word length in each document
    rumor_lens = [len(sentence) for sentence in rumor_corpus]
    doc_lens = [len(sentence) for sentence in doc_corpus]

    print "mean length of words, (rumor: %f, doc: %f)" % \
          (np.mean(rumor_lens), np.mean(doc_lens))
    print "std length of words, (rumor: %f, doc: %f)" % \
          (np.std(rumor_lens), np.std(doc_lens))

    # plot hist
    plt.style.use('ggplot')
    plt.hist([rumor_lens, doc_lens], 20, histtype='bar',
             cumulative=True, normed=1, label=['rumor', 'doc'])
    plt.legend()
    plt.xlabel("words per text")
    plt.ylabel("Probability")
    plt.title("Distribution of Rumor and Doc Text Length (Words)")
    plt.show()
