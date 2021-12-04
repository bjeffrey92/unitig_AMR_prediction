import pickle
import re
import glob
import logging
import os
from typing import Dict, Union

import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from numpy import corrcoef

from linear_model.utils import (
    ResultsContainer,
    load_training_data,
    load_adjacency_matrix,
    load_metadata,
)
from linear_model.lasso_model import fit_model
from linear_model.elastic_net import fit_model as fit_elastic_net


def get_results(results_dir: str) -> Dict:
    files = glob.glob(results_dir + "/*predictions.pkl")
    results = {}
    for f in files:
        Ab = re.findall(r"log2_(.*)_mic", f)[0]
        with open(f, "rb") as a:
            results[Ab] = pickle.load(a)
    return {k: results[k] for k in sorted(results.keys())}


def extract_optimal_hyperparams(ab_results: Dict) -> Dict:
    clade_optimal_hyperparams = {}
    for clade, data in ab_results.items():
        if isinstance(data, list):
            test_accuracies = [i.testing_accuracy for i in data]
            best_fit_ix = test_accuracies.index(max(test_accuracies))
            clade_optimal_hyperparams[clade] = data[best_fit_ix].hyperparameters
        elif isinstance(data, ResultsContainer):
            clade_optimal_hyperparams[clade] = data.hyperparameters
        else:
            raise TypeError(type(data))
    return clade_optimal_hyperparams


def plot_best_fits(
    results: Dict,
    optimal_hyperparams: Dict = None,
    filename: str = None,
):
    """
    results and optimal hyperparameters are nested dictionaries containing the
    data for each Ab and clade
    """
    # Extract metrics of accuracy for best fitting model
    Ab_accuracies = {}
    for Ab, Ab_data in results.items():

        training_accuracy = []
        testing_accuracy = []
        validation_accuracy = []
        for clade, clade_data in Ab_data.items():
            if isinstance(clade_data, ResultsContainer) and optimal_hyperparams is None:
                best_model_results = clade_data
            elif optimal_hyperparams is not None:
                best_model_results = next(
                    filter(
                        lambda x: x.hyperparameters == optimal_hyperparams[Ab][clade],
                        clade_data,
                    )
                )
            else:
                raise ValueError(
                    "If optimal hyperparameters are not supplied then results must be a dictionary of ResultsContainers"  # noqa: E501
                )
            training_accuracy.append(best_model_results.training_accuracy)
            testing_accuracy.append(best_model_results.testing_accuracy)
            validation_accuracy.append(best_model_results.validation_accuracy)

        Ab_accuracies[Ab] = pd.DataFrame(
            {
                "Train": training_accuracy,
                "Test": testing_accuracy,
                "Validate": validation_accuracy,
            },
            index=list(Ab_data.keys()),
        )

    plt.clf()
    # Plot train test and validate
    fig, axs = plt.subplots(1, 4, sharey=True)
    n = 0
    for Ab in Ab_accuracies.keys():
        df = Ab_accuracies[Ab]
        axs[n].boxplot(df, notch=False, labels=df.columns)
        axs[n].set_title(Ab.upper())
        axs[n].tick_params(labelrotation=90)
        axs[n].set_ylim([0, 100])
        n += 1

    fig.text(0, 0.5, "Mean Accuracy per MIC Bin", va="center", rotation="vertical")
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()


def fit_Ab_models(
    Ab: str,
    Ab_hps: Dict[int, Dict[str, Union[float, int]]],
    model_type: str,
    data_dir: str,
) -> Dict[int, Union[Lasso, Ridge, ElasticNet]]:

    training_data = load_training_data(data_dir)
    training_metadata = load_metadata(data_dir)[0]

    models = {}
    for clade, hps in Ab_hps.items():
        training_indices = training_metadata.loc[
            training_metadata.clusters != clade
        ].index
        training_features = torch.index_select(
            training_data[0], 0, torch.as_tensor(training_indices)
        )
        training_labels = torch.index_select(
            training_data[1], 0, torch.as_tensor(training_indices)
        )

        logging.info(f"Fitting model for {Ab} and clade: {clade}")
        if model_type == "elastic_net":
            model = fit_elastic_net(training_features, training_labels, **hps)
        else:
            model = fit_model(training_features, training_labels, model_type, **hps)
        models[clade] = model

    return models


def evaluate_models(
    Ab: str,
    models_dict: Dict[int, Union[Lasso, Ridge, ElasticNet]],
    data_dir: str,
    results_dir: str,
):
    adj = load_adjacency_matrix(data_dir)
    indices = pd.DataFrame(
        adj.indices().transpose(0, 1).tolist()
    )  # parse edges as dataframe for quick searching
    indices = indices.loc[indices[0] != indices[1]]  # remove self loops

    corr_coefs = {}
    fig, axs = plt.subplots(4, 3, sharex=True, sharey=True)
    n = 0
    m = 0
    for clade, model in models_dict.items():
        x_list = []
        y_list = []
        if isinstance(model, ResultsContainer):
            model = model.model
        for i in range(model.coef_.shape[0]):
            y_ix = indices.loc[indices[0] == i][1]  # get neighbours of i
            y_list += model.coef_[y_ix].tolist()
            x_list += [model.coef_[i].tolist()] * len(y_ix)

        axs[n, m].scatter(x_list, y_list, s=1)
        axs[n, m].set_title(clade)
        if n > 0 and n % 3 == 0:
            n = 0
            m += 1
        else:
            n += 1

        corr_coefs[clade] = corrcoef(x_list, y_list)[
            0, 1
        ]  # pearson correlation coefficient

    fig.tight_layout()
    fig.savefig(
        os.path.join(results_dir, f"{Ab}_neighbouring_nodes_model_coefficients.png")
    )
    with open(
        os.path.join(
            results_dir,
            f"{Ab}_neighbouring_nodes_model_coefficients_pearson_coef.pkl",
        ),
        "wb",
    ) as a:
        pickle.dump(corr_coefs, a)


def convolution_impact(Ab: str, results_dir: str):
    # get the corr coefficients for the convolved and non-convolved data
    pearson_coefficients = []
    with open(
        os.path.join(
            results_dir,
            f"{Ab}_neighbouring_nodes_model_coefficients_pearson_coef.pkl",
        ),
        "rb",
    ) as a:
        pearson_coefficients.append(pickle.load(a))
    with open(
        os.path.join(
            results_dir,
            "convolved",
            f"{Ab}_neighbouring_nodes_model_coefficients_pearson_coef.pkl",
        ),
        "rb",
    ) as a:
        pearson_coefficients.append(pickle.load(a))
    # unpack dictionaries of each clade
    pearson_coefficients = [list(i.values()) for i in pearson_coefficients]

    plt.clf()
    plt.boxplot(pearson_coefficients, labels=["Unconvolved", "Convolved"])
    plt.tick_params(labelrotation=90)
    plt.ylabel("Pearson Correlation Coefficient")
    plt.title(Ab.upper())
    plt.tight_layout()


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    for model_type in ["ridge", "lasso", "elastic_net"]:
        results_dir = f"linear_model/results/{model_type}_results/gwas_filtered/cluster_wise_CV"  # noqa: E501
        root_dir = "data/gonno/model_inputs/freq_5_95/"  # to access raw data

        results = get_results(results_dir)

        if model_type == "elastic_net":  # hps optimised with bayes_opt
            for Ab, models_dict in results.items():
                data_dir = os.path.join(root_dir, f"log2_{Ab}_mic", "gwas_filtered")
                evaluate_models(Ab, models_dict, data_dir, results_dir)

        else:  # others did grid search over alpha
            optimal_hyperparams = {
                Ab: extract_optimal_hyperparams(Ab_results)
                for Ab, Ab_results in results.items()
            }

            plot_best_fits(
                results,
                optimal_hyperparams,
                os.path.join(results_dir, "best_model_accuracies.png"),
            )

            for Ab, Ab_hps in optimal_hyperparams.items():
                data_dir = os.path.join(root_dir, f"log2_{Ab}_mic", "gwas_filtered")

                models_dict = fit_Ab_models(Ab, Ab_hps, model_type, data_dir)

                evaluate_models(Ab, models_dict, data_dir, results_dir)
