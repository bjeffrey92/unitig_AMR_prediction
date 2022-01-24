import pickle
import os
import logging
from typing import Dict
from functools import partial

import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from skranger.ensemble import RangerForestRegressor
from xgboost import XGBRegressor

from linear_model.utils import (
    load_training_data,
    load_testing_data,
    load_metadata,
    load_adjacency_matrix,
    load_adjacency_dictionary,
    train_test_validate_split,
    convolve,
    mean_acc_per_bin,
    ResultsContainer,
)

# from .julia_interface import get_jl_decision_tree, graph_rf_model
from decision_tree_models.julia_interface import (
    get_jl_modules,
    graph_rf_model,
    julia_rf_model,
)

# from .utils import convert_adj_matrix, adj_matrix_to_dict
from decision_tree_models.utils import adj_matrix_to_dict

JL_ENV_PATH = (
    "/home/bj515/OneDrive/work_stuff/WGS_AMR_prediction/graph_learning/DecisionTree.jl"
)
DecisionTree, JLD = get_jl_modules(JL_ENV_PATH)


def fit_graph_rf(training_features, training_labels, adj, **kwargs) -> graph_rf_model:
    if not isinstance(adj, dict):
        adj = adj_matrix_to_dict(adj)
    reg = graph_rf_model(
        DecisionTree,
        JLD,
        training_features,
        training_labels,
        adj,
        **kwargs,
    )
    reg.fit()
    return reg


def fit_julia_rf(training_features, training_labels, **kwargs) -> julia_rf_model:
    reg = julia_rf_model(
        DecisionTree, JLD, training_features, training_labels, **kwargs
    )
    reg.fit()
    return reg


def fit_xgboost(training_features, training_labels, **kwargs) -> XGBRegressor:
    kwargs = {k: round(v) for k, v in kwargs.items()}
    reg = XGBRegressor(**kwargs)
    reg.fit(training_features, training_labels)
    return reg


def fit_rf(training_features, training_labels, **kwargs) -> RangerForestRegressor:
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
    convolve_features: bool = False,
    **kwargs,
):

    if convolve_features:
        training_features = convolve(training_features, adj)
        testing_features = convolve(testing_features, adj)
        if validation_features is not None:
            validation_features = convolve(validation_features, adj)

    # keep as tensor so can be cached (needs to be hashable)
    training_features = np.array(training_features)
    training_labels = np.array(training_labels)
    testing_features = np.array(testing_features)
    testing_labels = np.array(testing_labels)
    if validation_features is not None:
        validation_features = np.array(validation_features)
        validation_labels = np.array(validation_labels)

    if model_type == "random_forest":
        reg = fit_rf(training_features, training_labels, **kwargs)
    elif model_type == "xgboost":
        reg = fit_xgboost(training_features, training_labels, **kwargs)
    elif model_type == "graph_rf":
        reg = fit_graph_rf(training_features, training_labels, adj, **kwargs)
    elif model_type == "julia_rf":
        reg = fit_julia_rf(training_features, training_labels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if validation_features is None:
        testing_predictions = reg.predict(testing_features)
        testing_loss = float(mean_squared_error(testing_labels, testing_predictions))
        return -testing_loss
    else:
        training_predictions = reg.predict(training_features)
        testing_predictions = reg.predict(testing_features)
        validation_predictions = reg.predict(validation_features)

        training_accuracy = mean_acc_per_bin(training_predictions, training_labels)
        testing_accuracy = mean_acc_per_bin(testing_predictions, testing_labels)
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
            training_MSE=mean_squared_error(training_labels, training_predictions),
            testing_MSE=mean_squared_error(testing_labels, testing_predictions),
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
    convolve_features: bool = False,
    n_splits: int = 12,
) -> Dict:
    clades = np.sort(training_metadata.clusters.unique())
    assert np.array_equal(
        np.sort(clades), np.sort(testing_metadata.clusters.unique())
    ), "Different clades found in training and testing metadata"

    n_splits = max([n_splits, len(clades)])
    clade_groups = [list(i) for i in np.array_split(clades, n_splits)]
    clade_groups = [i for i in clade_groups if len(i) > 0]

    results_dict = {}
    for left_out_clade in clade_groups:
        logging.info(f"Formatting data for model with clades {left_out_clade} left out")
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
            pbounds = {
                "max_depth": [2, 5],
                "n_estimators": [1000, 10000],
                "eta": [0, 1],  # learning rate
                "gamma": [0, 10],  # min loss needed to split the tree further
                "min_child_weight": [2, 10],  # samples per node
                "lambda": [0, 2],  # l2 regularization constant
                "alpha": [0, 2],  # l1 regularlization constant
            }
        elif model_type in ["graph_rf", "julia_rf"]:
            pbounds = {
                "n_trees": [10, 1000],
                "max_depth": [5, 20],
                "min_samples_split": [5, 25],
                "min_purity_increase": [0.01, 0.3],
            }
            if model_type == "graph_rf":
                pbounds = {
                    **pbounds,
                    "jump_probability": [0.01, 0.2],
                    "graph_steps": [1, 6],
                }
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
            convolve_features=convolve_features,
        )

        logging.info("Optimizing hyperparameters")
        optimizer = BayesianOptimization(
            f=partial_fitting_function, pbounds=pbounds, random_state=1
        )
        if model_type == "random_forest":
            optimizer.maximize(init_points=5, n_iter=5)
        elif model_type == "xgboost":
            optimizer.maximize(init_points=10, n_iter=20)
        elif model_type in ["graph_rf", "julia_rf"]:
            optimizer.maximize(init_points=8, n_iter=15)
        else:
            raise ValueError(f"Unknown model type {model_type}")

        logging.info(
            "Optimization complete, extracting metrics for best hyperparameter \
        combination"
        )
        results_dict[str(left_out_clade)] = train_evaluate(
            *(list(input_data) + [adj, model_type]),
            **optimizer.max["params"],
        )

    return results_dict


def save_output(results_dict: Dict, results_dir: str, outcome: str, model_type: str):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # cant pickle julia object so call method to save model as jld object and record
    # the path in the pickled python object
    if model_type in ["graph_rf", "julia_rf"]:
        for k, v in results_dict.items():
            clade_fname = os.path.join(
                results_dir, outcome + f"_clade_{k}_{model_type}.jld"
            )
            v.model.save_model(clade_fname)
            v.model = clade_fname
            results_dict[k] = v

    fname = os.path.join(results_dir, outcome + f"_CV_{model_type}_predictions.pkl")
    with open(fname, "wb") as a:
        pickle.dump(results_dict, a)


def main(
    outcome: str,
    root_dir: str,
    model_type: str = "random_forest",
    convolve: bool = False,
    gwas_filtered: bool = True,
    results_dir_suffix: str = "",
):

    logging.info(f"Fitting models with {outcome}")
    data_dir = os.path.join(root_dir, outcome)
    if gwas_filtered:
        data_dir = os.path.join(data_dir, "gwas_filtered")
        results_dir = os.path.join(
            f"decision_tree_models/results/{model_type}/gwas_filtered/",
            "cluster_wise_CV",
        )
    else:
        results_dir = os.path.join(
            f"decision_tree_models/results/{model_type}", "cluster_wise_CV"
        )
    if convolve:
        results_dir = os.path.join(results_dir, "convolved")

    if results_dir.endswith("/"):
        results_dir = results_dir[:-1]
    results_dir += results_dir_suffix

    training_data = load_training_data(data_dir)
    testing_data = load_testing_data(data_dir)
    training_metadata, testing_metadata = load_metadata(data_dir)
    if model_type == "graph_rf":
        adj = load_adjacency_dictionary(data_dir)
    elif convolve:
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
        convolve_features=convolve,
    )

    save_output(results_dict, results_dir, outcome, model_type)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    root_dir = "data/gonno/model_inputs/unfiltered"
    # root_dir = "data/gonno/model_inputs/freq_5_95/"
    outcomes = os.listdir(root_dir)
    for outcome in outcomes:
        main(outcome, root_dir)
