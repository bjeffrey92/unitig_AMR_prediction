import os
from math import log2
from typing import Tuple

import ingot
import numpy as np
from nptyping import NDArray
from sklearn.metrics import balanced_accuracy_score

from GNN_model.utils import breakpoints
from linear_model.utils import (
    convolve as convolve_,
    load_adjacency_matrix,
    load_testing_data,
    load_training_data,
)


ROOT_DIR = "data/gonno/model_inputs/freq_5_95/"


def load_data(
    outcome: str, *, train_or_test: str = "train", convolve: bool = False
) -> Tuple[NDArray, NDArray]:
    data_dir = os.path.join(ROOT_DIR, outcome, "gwas_filtered")

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


def main(outcome: str, convolve: bool = False):
    train_X, train_y = load_data(outcome, convolve=convolve)
    clf = ingot.INGOTClassifier(
        lambda_p=10,
        lambda_z=0.01,
        false_positive_rate_upper_bound=0.1,
        max_rule_size=20,
        solver_name="CPLEX_PY",
        solver_options={"timeLimit": 1800},
    )
    clf.fit(train_X, train_y)

    test_X, test_y = load_data(
        outcome, train_or_test="test", convolve=convolve
    )
    test_pred = clf.predict(test_X)
    balanced_accuracy_score(test_y, test_pred)


if __name__ == "__main_":
    outcomes = os.listdir(ROOT_DIR)
    main(outcomes[0])
