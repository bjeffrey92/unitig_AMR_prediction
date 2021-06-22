import os
import pickle
import subprocess
from math import log2
from typing import Tuple

import ingot
import numpy as np
import pandas as pd
from nptyping import NDArray
from scipy.sparse import csr_matrix
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


def _generate_reduced_features(
    train_X: NDArray,
    train_y: NDArray,
    output_filename: str,
    geno_filename: str,
    pheno_filename: str,
) -> Tuple[NDArray, NDArray, NDArray]:

    if not os.path.isfile(output_filename):
        if any(
            np.concatenate(
                [
                    np.apply_along_axis(lambda x: all(x == 0), 0, train_X),
                    np.apply_along_axis(lambda x: all(x == 0), 1, train_X),
                ]
            )
        ):
            raise ValueError("Empty rows or columns in genotype matrix")

        uniq_matrix, uniq_indices = np.unique(
            train_X, return_index=True, axis=0
        )
        if uniq_matrix.shape != train_X.shape:
            uniq_indices = sorted(uniq_indices)
            train_X = np.stack([train_X[i] for i in uniq_indices])
            train_y = np.array([train_y[i] for i in uniq_indices])

        with open(pheno_filename, "a") as a:
            # add 1 for R
            for i in ["row"] + (np.where(train_y == 1)[0] + 1).tolist():
                a.write(str(i) + "\n")

        features_dict = {}
        for i, j in enumerate(train_X):
            # add 1 for R
            features = np.where(j == 1)[0] + 1
            features_dict[i + 1] = (
                np.array([i + 1] * len(features)),
                features,
            )
        geno_df = pd.DataFrame(
            {
                "row": np.concatenate([i[0] for i in features_dict.values()]),
                "col": np.concatenate([i[1] for i in features_dict.values()]),
            }
        )
        geno_df.to_csv(geno_filename, index=False)

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = "/home/bj515/OneDrive/work_stuff/WGS_AMR_prediction/graph_learning/ingot_dr"  # noqa: E501

        r_script = os.path.join(
            base_dir, "feature_reduction/FeatureReduction.R"
        )
        status = subprocess.call(
            [
                "Rscript",
                r_script,
                "-g",
                geno_filename,
                "-p",
                pheno_filename,
                "-o",
                output_filename,
            ]
        )
        if status != 0:
            raise Exception("Building reduced feature file failed")

    reduced_features = pd.read_csv(output_filename) - 1
    sparse_reduced_features = csr_matrix(
        (
            [1] * len(reduced_features),
            (reduced_features.row, reduced_features.col),
        )
    )
    reduced_train_X = np.array(sparse_reduced_features.todense())
    # store indices to filter the testing data
    feature_indices = np.unique(reduced_features.col)

    return reduced_train_X, train_y, feature_indices


def reduce_features(
    train_X: NDArray,
    train_y: NDArray,
    output_dir: str,
    file_prefix: str,
):
    reduced_data_filename = os.path.join(
        output_dir, f"{file_prefix}_reduced_features.pkl"
    )
    reduced_features_filename = os.path.join(
        output_dir, f"{file_prefix}_reduced_features.csv"
    )
    geno_filename = os.path.join(output_dir, f"{file_prefix}_genotype.csv")
    pheno_filename = os.path.join(output_dir, f"{file_prefix}_phenotype.csv")

    if not os.path.isfile(reduced_data_filename):
        train_X, train_y, feature_indices = _generate_reduced_features(
            train_X,
            train_y,
            reduced_features_filename,
            geno_filename,
            pheno_filename,
        )
        with open(reduced_data_filename, "wb") as a:
            pickle.dump((train_X, train_y, feature_indices), a)
    else:
        with open(reduced_data_filename, "rb") as a:
            train_X, train_y, feature_indices = pickle.load(a)

    return train_X, train_y, feature_indices


def main(outcome: str, convolve: bool = False, reduce: bool = False):

    if convolve and reduce:
        raise ValueError("Only one of convolve or reduce can be True")

    train_X, train_y = load_data(outcome, convolve=convolve)
    test_X, test_y = load_data(
        outcome, train_or_test="test", convolve=convolve
    )
    if reduce:
        train_X, train_y, feature_indices = reduce_features(
            train_X,
            train_y,
            "ingot_dr/reduced_unitig_features/",
            outcome,
        )

    clf = ingot.INGOTClassifier(
        lambda_p=10,
        lambda_z=0.01,
        false_positive_rate_upper_bound=0.1,
        max_rule_size=20,
        solver_name="CPLEX_PY",
        solver_options={"timeLimit": 1800},
    )
    clf.fit(train_X, train_y)

    test_pred = clf.predict(test_X)
    balanced_accuracy_score(test_y, test_pred)


if __name__ == "__main_":
    outcomes = os.listdir(ROOT_DIR)
    main(outcomes[0])
