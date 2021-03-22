import math
from typing import Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def essential_agreement(df: pd.DataFrame) -> float:
    diff = abs(df.predictions - df.labels)
    correct = diff[[i < 1 for i in diff]]
    return len(correct) / len(df) * 100


def acc_per_bin(
    predictions: pd.Series, labels: pd.Series, bin_size="optimise"
) -> Tuple[List[float], List[int], int]:
    assert len(predictions) == len(labels)

    if type(bin_size) != int and bin_size != "optimise":
        raise ValueError("bin_size must = optimise, or be an integer >= 1 ")
    if type(bin_size) == int and bin_size < 1:
        raise NotImplementedError("If an integer, bin_size must be >= 1")

    # apply Freedman-Diaconis rule to get optimal bin size
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    if bin_size == "optimise":
        IQR = np.subtract(*np.percentile(labels, [75, 25]))
        bin_size = 2 * IQR / (len(labels) ** (1 / 3))
        bin_size = int(
            np.ceil(bin_size)
        )  # round up cause if less than 1 will not work with accuracy function

    min_value = int(np.floor(min(labels)))
    max_value = int(np.floor(max(labels)))
    bins = list(range(min_value, max_value + bin_size, bin_size))
    binned_labels = np.digitize(labels, bins)

    # centroid of each bin for plotting
    bins = [i + bin_size / 2 for i in bins][:-1]

    df = pd.DataFrame(
        {
            "labels": labels,
            "predictions": predictions,
            "binned_labels": binned_labels,
        }
    )  # to allow quick searches across bins
    bin_accuracies = df.groupby(df.binned_labels).apply(essential_agreement)

    return bin_accuracies.to_list(), bins, bin_size


def plot_incorrect_by_MIC_bin(labels: pd.Series, predictions: pd.Series):
    assert len(labels) == len(predictions)

    # remove missing values
    predictions = predictions[~pd.isna(labels)]
    labels = labels[~pd.isna(labels)]

    # binary log
    labels = labels.apply(math.log2)
    predictions = predictions.apply(math.log2)

    bin_accuracies, bins, bin_size = acc_per_bin(predictions, labels)

    plt.clf()
    plt.bar(bins, bin_accuracies, width=bin_size - (bin_size * 0.1))
    plt.axvline(math.log2(0.06), color="black")
    plt.axvline(math.log2(2), color="black")
    plt.xlabel("log2(MIC)")
    plt.ylabel("Prediction Accuracy per Bin")


if __name__ == "__main__":
    Ab = "PEN"

    d1_file = "data/pneumo_pbp/li_et_al_data/12864_2017_4017_MOESM1_ESM.csv"
    d2_file = "data/pneumo_pbp/li_et_al_data/12864_2017_4017_MOESM2_ESM.csv"

    d1 = pd.read_csv(d1_file)
    d2 = pd.read_csv(d2_file)

    plot_incorrect_by_MIC_bin(d1[Ab], d1[f"{Ab}_MIC_RF"])
