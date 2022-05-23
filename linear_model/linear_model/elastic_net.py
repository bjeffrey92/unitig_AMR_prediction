import pickle
import os
import logging
import warnings
from functools import partial

import numpy as np
from numpy import sort, array_equal
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

from linear_model.utils import (
    load_training_data,
    load_testing_data,
    load_metadata,
    load_adjacency_matrix,
    train_test_validate_split,
    mean_acc_per_bin,
    ResultsContainer,
    convolve,
)


def fit_model(training_features, training_labels, alpha, l1_ratio):
    max_iter = 50000
    fitted = False
    while not fitted:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            reg = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=0,
                max_iter=max_iter,
            )
            reg.fit(training_features, training_labels)

            if len(w) > 1:
                for warning in w:
                    logging.error(warning.category)
                raise Exception
            elif w and issubclass(w[0].category, ConvergenceWarning):
                logging.warning(
                    f"Failed to converge with max_iter = {max_iter}, \
                adding 100000 more"
                )
                max_iter += 50000
            else:
                fitted = True

    return reg


def train_evaluate(
    training_features,
    training_labels,
    testing_features,
    testing_labels,
    validation_features,
    validation_labels,
    adj,
    alpha,
    l1_ratio,
):

    logging.info(f"alpha = {alpha}, l1_ratio = {l1_ratio}")

    if adj is not None:
        training_features = convolve(training_features, adj)
        testing_features = convolve(testing_features, adj)
        if validation_features is not None:
            validation_features = convolve(validation_features, adj)

    reg = fit_model(training_features, training_labels, alpha, l1_ratio)

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
            hyperparameters={"alpha": alpha, "l1_ratio": l1_ratio},
            model_type="elastic_net",
            model=reg,
        )

        return results


def leave_one_out_CV(
    training_data,
    testing_data,
    training_metadata,
    testing_metadata,
    adj=None,
    n_splits=6,
):

    try:
        clades = np.sort(training_metadata.clusters.unique())
    except AttributeError:
        logging.warning(
            "clusters column not found in metadata, attempting to use 'Clade' instead"
        )
        training_metadata = training_metadata.assign(clusters=training_metadata.Clade)
        testing_metadata = testing_metadata.assign(clusters=testing_metadata.Clade)
        clades = np.sort(training_metadata.clusters.unique())

    assert array_equal(
        sort(clades), sort(testing_metadata.clusters.unique())
    ), "Different clades found in training and testing metadata"

    clade_groups = [list(i) for i in np.array_split(clades, n_splits)]
    clade_groups = [i for i in clade_groups if len(i) > 0]

    results_dict = {}
    for left_out_clade in clade_groups:
        logging.info(f"Formatting data for model with clade {left_out_clade} left out")
        input_data = train_test_validate_split(
            training_data,
            testing_data,
            training_metadata,
            testing_metadata,
            left_out_clade,
            torch_or_numpy="numpy",
        )

        (
            training_features,
            training_labels,
            testing_features,
            testing_labels,
        ) = input_data[:4]

        pbounds = {"alpha": (0.01, 0.1), "l1_ratio": (0.3, 0.7)}

        partial_fitting_function = partial(
            train_evaluate,
            training_features=training_features,
            training_labels=training_labels,
            testing_features=testing_features,
            testing_labels=testing_labels,
            validation_features=None,
            validation_labels=None,
            adj=adj,
        )

        logging.info("Optimizing hyperparameters")
        optimizer = BayesianOptimization(
            f=partial_fitting_function, pbounds=pbounds, random_state=1
        )
        optimizer.maximize(init_points=3, n_iter=3)

        logging.info(
            "Optimization complete, extracting metrics for best hyperparameter \
        combination"
        )
        results_dict[str(left_out_clade)] = train_evaluate(
            *input_data,
            adj,
            optimizer.max["params"]["alpha"],
            optimizer.max["params"]["l1_ratio"],
        )

    return results_dict


def save_output(results_dict, results_dir, outcome):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fname = os.path.join(results_dir, outcome + "_CV_elastic_net_predictions.pkl")
    with open(fname, "wb") as a:
        pickle.dump(results_dict, a)


def main(species, root_dir, convolve=False, results_dir_suffix=""):
    outcomes = os.listdir(root_dir)
    for outcome in outcomes:
        logging.info(f"Fitting models with {outcome}")
        results_dir = (
            f"linear_model/results/{species}elastic_net_results/cluster_wise_CV"
        )
        data_dir = os.path.join(root_dir, outcome)
        # results_dir = (
        #     "linear_model/results/elastic_net_results/gwas_filtered/"
        #     + "cluster_wise_CV"
        # )
        if convolve:
            results_dir = os.path.join(results_dir, "convolved")

        if results_dir.endswith("/"):
            results_dir = results_dir[:-1]
        results_dir += results_dir_suffix

        training_data = load_training_data(data_dir)
        testing_data = load_testing_data(data_dir)
        training_metadata, testing_metadata = load_metadata(data_dir)

        # fitting elastic net is more efficient with fortran contigous array
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet.fit
        training_data = [np.array(i, order="F") for i in training_data]
        testing_data = [np.array(i, order="F") for i in testing_data]

        if convolve:
            adj = load_adjacency_matrix(data_dir)
        else:
            adj = None

        results_dict = leave_one_out_CV(
            training_data,
            testing_data,
            training_metadata,
            testing_metadata,
            adj=adj,
        )

        save_output(results_dict, results_dir, outcome)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    root_dir = "data/euscape/model_inputs/"
    species = "kleb"
    main(species, root_dir)
