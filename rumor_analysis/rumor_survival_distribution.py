# -*- encoding: utf-8 -*-
# author: Qiaoan Chen <kazenoyumechen@gmail.com>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def convert_to_category(duration):
    if duration <= 0:
        return np.nan
    elif duration < 86400:
        return "1 day"
    elif duration < 259200:
        return "1~3 day"
    elif duration < 604800:
        return "3 day ~ 1 week"
    elif duration < 1209600:
        return "1 week ~ 2 week"
    else:
        return "> 2 week"


if __name__ == "__main__":
    """
    Analzing the survival time distribution of rumors.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "rumor_path",
        help="rumor csv file path"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.rumor_path)
    df["duration"] = df["punishtime"] - df["post_time"]
    df["duration_cat"] = [convert_to_category(duration) for duration in df["duration"]]

    plt.style.use('ggplot')
    indices = ["1 day", "1~3 day", "3 day ~ 1 week", "1 week ~ 2 week", "> 2 week"]
    df["duration_cat"].value_counts()[indices].plot(kind='bar')
    plt.xlabel("rumor survival time")
    plt.ylabel("count")
    plt.title("Histogram of Rumor Survival Time")
    plt.show()
