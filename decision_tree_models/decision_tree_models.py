import pickle
import os
import logging
from typing import Dict
from functools import partial

import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from skranger.ensemble import RangerForestRegressor

from linear_model.utils import (
    load_training_data,
    load_testing_data,
    load_metadata,
    load_adjacency_matrix,
    train_test_validate_split,
    convolve,
    mean_acc_per_bin,
    ResultsContainer,
)


ROOT_DIR = "data/gonno/model_inputs/freq_5_95/"


def fit_xgboost(training_features, training_labels):
    ...


def fit_rf(
    training_features, training_labels, **kwargs
) -> RangerForestRegressor:
    kwargs = {k: round(v) for k, v in kwargs.items()}
    reg = RangerForestRegressor(**kwargs)
    reg.fit(training_features, training_labels)
    return reg


def train_evaluate(
    training_features,
    training_labels,
    testing_features,
    testing_labels,
    validation_features,
    validation_labels,
    adj: bool,
    model_type: str,
    **kwargs,
):

    if adj is not None:
        training_features = convolve(training_features, adj)
        testing_features = convolve(testing_features, adj)
        if validation_features is not None:
            validation_features = convolve(validation_features, adj)

    if model_type == "random_forest":
        reg = fit_rf(training_features, training_labels, **kwargs)
    elif model_type == "xgboost":
        raise NotImplementedError(model_type)
        # reg = fit_xgboost(training_features, training_labels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if validation_features is None:
        testing_predictions = reg.predict(testing_features)
        testing_loss = float(
            mean_squared_error(testing_labels, testing_predictions)
        )
        return -testing_loss
    else:
        training_predictions = reg.predict(training_features)
        testing_predictions = reg.predict(testing_features)
        validation_predictions = reg.predict(validation_features)

        training_accuracy = mean_acc_per_bin(
            training_predictions, training_labels
        )
        testing_accuracy = mean_acc_per_bin(
            testing_predictions, testing_labels
        )
        validation_accuracy = mean_acc_per_bin(
            validation_predictions, validation_labels
        )

        logging.info(
            {
                "training_accuracy": training_accuracy,
                "testing_accuracy": testing_accuracy,
                "validation_accuracy": validation_accuracy,
            }
        )

        results = ResultsContainer(
            training_accuracy=training_accuracy,
            testing_accuracy=testing_accuracy,
            validation_accuracy=validation_accuracy,
            training_MSE=mean_squared_error(
                training_labels, training_predictions
            ),
            testing_MSE=mean_squared_error(
                testing_labels, testing_predictions
            ),
            validation_MSE=mean_squared_error(
                validation_labels, validation_predictions
            ),
            training_predictions=training_predictions,
            testing_predictions=testing_predictions,
            validation_predictions=validation_predictions,
            hyperparameters=kwargs,
            model_type=model_type,
            model=reg,
        )

        return results


def leave_one_out_CV(
    training_data,
    testing_data,
    training_metadata,
    testing_metadata,
    model_type: str = "random_forest",
    adj: str = None,
) -> Dict:
    clades = np.sort(training_metadata.clusters.unique())
    assert np.array_equal(
        np.sort(clades), np.sort(testing_metadata.clusters.unique())
    ), "Different clades found in training and testing metadata"

    results_dict = {}
    for left_out_clade in clades:
        logging.info(
            f"Formatting data for model with clade {left_out_clade} left out"
        )
        input_data = train_test_validate_split(
            training_data,
            testing_data,
            training_metadata,
            testing_metadata,
            left_out_clade,
        )
        (
            training_features,
            training_labels,
            testing_features,
            testing_labels,
        ) = input_data[:4]

        if model_type == "random_forest":
            pbounds = {
                "n_estimators": [1000, 10000],
                "max_depth": [2, 5],
                "min_node_size": [2, 10],
            }
        elif model_type == "xgboost":
            pbounds = {}
        else:
            raise ValueError(f"Unknown model type {model_type}")

        partial_fitting_function = partial(
            train_evaluate,
            training_features=training_features,
            training_labels=training_labels,
            testing_features=testing_features,
            testing_labels=testing_labels,
            validation_features=None,
            validation_labels=None,
            adj=adj,
            model_type=model_type,
        )

        logging.info("Optimizing hyperparameters")
        optimizer = BayesianOptimization(
            f=partial_fitting_function, pbounds=pbounds, random_state=1
        )
        optimizer.maximize(init_points=5, n_iter=5)

        logging.info(
            "Optimization complete, extracting metrics for best hyperparameter \
        combination"
        )
        results_dict[left_out_clade] = train_evaluate(
            *(list(input_data) + [adj, model_type]),
            **optimizer.max["params"],
        )

    return results_dict


def save_output(
    results_dict: Dict, results_dir: str, outcome: str, model_type: str
):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fname = os.path.join(
        results_dir, outcome + f"_CV_{model_type}_predictions.pkl"
    )
    with open(fname, "wb") as a:
        pickle.dump(results_dict, a)


def main(
    outcome: str,
    model_type: str = "random_forest",
    convolve: bool = False,
):

    logging.info(f"Fitting models with {outcome}")
    data_dir = os.path.join(ROOT_DIR, outcome, "gwas_filtered")
    results_dir = (
        f"decision_tree_models/results/{model_type}/gwas_filtered/"
        + "cluster_wise_CV"
    )
    if convolve:
        results_dir = os.path.join(results_dir, "convolved")

    training_data = load_training_data(data_dir)
    testing_data = load_testing_data(data_dir)
    training_metadata, testing_metadata = load_metadata(data_dir)
    if convolve:
        adj = load_adjacency_matrix(data_dir)
    else:
        adj = None

    results_dict = leave_one_out_CV(
        training_data,
        testing_data,
        training_metadata,
        testing_metadata,
        model_type,
        adj,
    )

    save_output(results_dict, results_dir, outcome, model_type)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    outcomes = os.listdir(ROOT_DIR)
    for outcome in outcomes:
        main(outcome)
