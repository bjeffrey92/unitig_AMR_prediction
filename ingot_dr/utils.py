import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from math import log2
from typing import Dict, Tuple

import numpy as np
from nptyping import NDArray

from GNN_model.utils import breakpoints
from linear_model.utils import (
    convolve as convolve_,
    load_adjacency_matrix,
    load_testing_data,
    load_training_data,
)


@dataclass(unsafe_hash=True)
class ResultsContainer:
    training_balanced_accuracy: float
    testing_balanced_accuracy: float

    training_accuracy: float
    testing_accuracy: float

    training_predictions: NDArray
    testing_predictions: NDArray

    config: Dict

    date_time = datetime.now()

    def __repr__(self):
        return (
            "INGOT-CLASSIFIER\n"
            + f"model fit config: {self.config}, \n"
            + "\n"
            + "ACCURACY\n"
            + f"Training Data Balanced Accuracy: {self.training_balanced_accuracy}\n"  # noqa: E501
            + f"Testing Data Balanced Accuracy: {self.testing_balanced_accuracy}\n"  # noqa: E501
        )


def load_data(
    outcome: str,
    root_dir: str,
    *,
    train_or_test: str = "train",
    convolve: bool = False,
) -> Tuple[NDArray, NDArray]:
    data_dir = os.path.join(root_dir, outcome, "gwas_filtered")

    if train_or_test == "train":
        unitigs_X, unitigs_y = load_training_data(data_dir)
    elif train_or_test == "test":
        unitigs_X, unitigs_y = load_testing_data(data_dir)
    else:
        raise ValueError(train_or_test)

    if convolve:
        unitigs_X = convolve_(unitigs_X, load_adjacency_matrix(data_dir))

    # read in as pytorch tensors
    unitigs_X = np.array(unitigs_X)
    unitigs_y = np.array(unitigs_y)

    unitigs_y = unitigs_y >= log2(breakpoints[outcome.split("_")[1]])
    unitigs_y = unitigs_y.astype(int)

    return unitigs_X, unitigs_y


def save_results(
    results: ResultsContainer,
    outdir: str,
    outcome: str,
    reduced: bool,
    convolved: bool,
):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    filename = f"{outcome}_results.pkl"
    if convolved:
        filename = f"convolved_{filename}"
    if reduced:
        filename = f"reduced_{filename}"

    # dont overwrite existing results file
    file_path = os.path.join(outdir, filename)
    i = 1
    while os.path.isfile(file_path):
        split_path = file_path.split(".")
        path_minus_ext = "".join(split_path[:-1])
        if i > 1:
            path_minus_ext = path_minus_ext[:-3]  # remove brackets and number
        ext = split_path[-1]
        file_path = path_minus_ext + f"({i})." + ext
        i += 1

    with open(file_path, "wb") as a:
        pickle.dump(results, a)
