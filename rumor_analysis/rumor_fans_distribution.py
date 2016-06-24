# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

if __name__ == "__main__":
    """
    Analzing the fans number distribution of rumors.
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

    rumor_df = pd.read_csv(args.rumor_path)
    doc_df = pd.read_csv(args.doc_path)
    rumor_ffans = rumor_df["ffans_avg_num"]
    doc_ffans = doc_df["ffans_avg_num"]

    print "mean fans: rumor %.3f, doc %.3f" % (rumor_ffans.mean(), doc_ffans.mean())
    print "std fans: rumor %.3f, doc %.3f" % (rumor_ffans.std(), doc_ffans.std())

    # plot hist
    plt.style.use('ggplot')
    plt.hist([rumor_ffans, doc_ffans], 20, histtype='bar', range=(1, 1e5), normed=1, label=['rumor', 'doc'])
    plt.legend()
    plt.xlabel("ffans average number")
    plt.ylabel("Probability")
    plt.title("Distribution of Rumor and Doc Fans Number")
    plt.show()

