import logging
import os
import pickle
import subprocess
from typing import Tuple

import ingot
import numpy as np
import pandas as pd
from nptyping import NDArray
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from ingot_dr.utils import load_data, save_results, ResultsContainer

ROOT_DIR = "data/gonno/model_inputs/freq_5_95/"


def _generate_reduced_features(
    train_X: NDArray,
    train_y: NDArray,
    output_filename: str,
    geno_filename: str,
    pheno_filename: str,
) -> Tuple[NDArray, NDArray, NDArray]:

    if not os.path.isfile(output_filename):
        logging.info("Generating Reduced Features")

        if any(
            np.concatenate(
                [
                    np.apply_along_axis(lambda x: all(x == 0), 0, train_X),
                    np.apply_along_axis(lambda x: all(x == 0), 1, train_X),
                ]
            )
        ):
            raise ValueError("Empty rows or columns in genotype matrix")

        # remove any duplicated rows or columns
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
    train_sample_indices = np.unique(reduced_features.row)
    feature_indices = np.unique(reduced_features.col)
    reduced_train_X = train_X[np.ix_(train_sample_indices, feature_indices)]

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
        logging.info("Loading Pre-Generated Reduced Features")
        with open(reduced_data_filename, "rb") as a:
            train_X, train_y, feature_indices = pickle.load(a)

    return train_X, train_y, feature_indices


def main(outcome: str, convolve: bool = False, reduce: bool = False, **kwargs):

    if convolve and reduce:
        raise ValueError("Only one of convolve or reduce can be True")

    logging.info(f"Loading Data for outcome: {outcome}")
    train_X, train_y = load_data(outcome, ROOT_DIR, convolve=convolve)
    test_X, test_y = load_data(
        outcome, ROOT_DIR, train_or_test="test", convolve=convolve
    )
    if reduce:
        train_X, train_y, feature_indices = reduce_features(
            train_X,
            train_y,
            "ingot_dr/reduced_unitig_features/",
            outcome,
        )
        test_X = test_X[:, feature_indices]

    logging.info("Fitting Model")
    clf = ingot.INGOTClassifier(**kwargs)
    clf.fit(train_X, train_y)

    logging.info("Saving Output")
    train_pred = clf.predict(train_X)
    test_pred = clf.predict(test_X)

    results = ResultsContainer(
        training_balanced_accuracy=balanced_accuracy_score(
            train_y, train_pred
        ),
        testing_balanced_accuracy=balanced_accuracy_score(test_y, test_pred),
        training_accuracy=accuracy_score(train_y, train_pred),
        testing_accuracy=accuracy_score(test_y, test_pred),
        training_predictions=train_pred,
        testing_predictions=test_pred,
        config=kwargs.update(  # type: ignore
            {
                "reduce": reduce,
                "convolve": convolve,
            }
        ),
    )
    print(results)

    save_results(results, "ingot_dr/results", outcome, reduce, convolve)


if __name__ == "__main_":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    outcomes = os.listdir(ROOT_DIR)
    for outcome in outcomes:
        main(
            outcome,
            reduce=True,
            lambda_p=10,
            lambda_z=0.01,
            false_positive_rate_upper_bound=0.1,
            max_rule_size=20,
            solver_name="CPLEX_PY",
            solver_options={"timeLimit": 1800},
        )
        main(
            outcome,
            convolve=True,
            lambda_p=10,
            lambda_z=0.01,
            false_positive_rate_upper_bound=0.1,
            max_rule_size=20,
            solver_name="CPLEX_PY",
            solver_options={"timeLimit": 1800},
        )
