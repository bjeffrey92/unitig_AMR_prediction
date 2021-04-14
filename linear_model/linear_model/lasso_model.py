import pickle
import os
import logging
import warnings
from typing import List

import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from numpy import linspace, sort, array_equal

from linear_model.utils import (
    load_training_data,
    load_testing_data,
    load_metadata,
    load_adjacency_matrix,
    mean_acc_per_bin,
    train_test_validate_split,
    convolve,
    ResultsContainer,
)


def fit_model(training_features, training_labels, model, alpha):

    logging.info(f"Fitting model for alpha = {alpha}")

    max_iter = 100000
    fitted = False
    while not fitted:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            if model == "lasso":
                reg = Lasso(alpha=alpha, random_state=0, max_iter=max_iter)
            elif model == "ridge":
                reg = Ridge(alpha=alpha, random_state=0, max_iter=max_iter)
            else:
                raise NotImplementedError(model)
            reg.fit(training_features, training_labels)

            if len(w) > 1:
                for warning in w:
                    logging.error(warning.category)
                raise Exception
            elif w and issubclass(w[0].category, ConvergenceWarning):
                logging.warning(
                    f"Failed to converge with max_iter = {max_iter},"
                    + " adding 100000 more"
                )
                max_iter += 100000
            else:
                fitted = True

    return reg


def fit_model_by_grid_search(
    training_features,
    training_labels,
    testing_features,
    testing_labels,
    validation_features,
    validation_labels,
    alphas: List,
    adj=None,
    model="lasso",
) -> List[ResultsContainer]:
    """
    Training and testing data is random split of part of the tree,
    validation data is from a separate clade
    """

    accuracies = []

    if adj is not None:
        training_features = convolve(training_features, adj)
        testing_features = convolve(testing_features, adj)

    for a in alphas:

        reg = fit_model(training_features, training_labels, model, a)

        logging.info(f"alpha = {a} model fitted, generating predictions")

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

        accuracies.append(
            ResultsContainer(
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
                hyperparameters={"alpha": a},
                model_type=model,
                model=reg,
            )
        )

    return accuracies


def leave_one_out_CV(
    training_data,
    testing_data,
    training_metadata,
    testing_metadata,
    model="lasso",
    adj=None,
):

    clades = sort(training_metadata.clusters.unique())
    assert array_equal(
        sort(clades), sort(testing_metadata.clusters.unique())
    ), "Different clades found in training and testing metadata"

    alphas = linspace(0.01, 0.2, 10)

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

        accuracy_dict = fit_model_by_grid_search(
            *input_data,
            alphas,
            model=model,
            adj=adj,
        )
        results_dict[left_out_clade] = accuracy_dict

    return results_dict


def save_output(accuracy_dict, results_dir, fname):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(os.path.join(results_dir, fname), "wb") as a:
        pickle.dump(accuracy_dict, a)


def plot_results(accuracy_dict, fname):

    training_acc = [i["training_accuracy"] for i in accuracy_dict.values()]
    testing_acc = [i["testing_accuracy"] for i in accuracy_dict.values()]
    validation_acc = [i["validation_accuracy"] for i in accuracy_dict.values()]

    plt.clf()

    plt.scatter(
        list(accuracy_dict.keys()),
        training_acc,
        label="Training Data Accuracy",
    )
    plt.scatter(
        list(accuracy_dict.keys()), testing_acc, label="Testing Data Accuracy"
    )
    plt.scatter(
        list(accuracy_dict.keys()),
        validation_acc,
        label="Left Out Clade Accuracy",
    )
    plt.legend(loc="lower left")

    plt.savefig(fname)


def main(convolve=False):
    root_dir = "data/gonno/model_inputs/freq_5_95/"

    for model in ["lasso", "ridge"]:
        outcomes = os.listdir(root_dir)
        for outcome in outcomes:
            data_dir = os.path.join(root_dir, outcome, "gwas_filtered")
            results_dir = (
                f"linear_model/results/{model}_results/"
                + "gwas_filtered/cluster_wise_CV"
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
                model=model,
                adj=adj,
            )

            fname = outcome + f"_CV_{model}_predictions.pkl"
            save_output(results_dict, results_dir, fname)

        # for left_out_clade in results_dict.keys():
        #     fname = os.path.join(results_dir,
        #         outcome + \
        #         f'_validation_cluster_{left_out_clade}_lasso_predictions.png')
        #     plot_results(results_dict[left_out_clade], fname)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main()
