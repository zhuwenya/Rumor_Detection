# coding=utf-8
# author: Qiaoan Chen <kazenoyumechen@gmail.com>


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def get_categorical_name(fromtype):
    """
    Convert fromtype to its corresponding name.
    Input:
    - fromtype: int, message forward type.
    - name: message forward type corresponding name.
    """
    idx_to_name = {
        0: "未知",
        1: "朋友圈",
        2: "单聊",
        3: "群聊",
        4: "收藏",
        5: "APP分享",
        6: "微信全局搜索",
        7: "公众号搜索",
        8: "公众号热词搜索",
        9: "公众号分享"
    }
    name = idx_to_name[fromtype]
    return name


if __name__ == "__main__":
    """
    Analzing forward type of messages distribution.
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
    rumor_df["fromtype_cat"] = map(get_categorical_name, rumor_df["fromtype"])
    doc_df["fromtype_cat"] = map(get_categorical_name, doc_df["fromtype"])


    # print value count
    print "rumor:"
    print rumor_df["fromtype_cat"].value_counts()

    print "document:"
    print doc_df["fromtype_cat"].value_counts()

