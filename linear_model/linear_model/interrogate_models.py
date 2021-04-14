import pickle
import re
import glob
import logging
import os
from typing import Dict, Union

import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import Lasso, Ridge

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
    return results


def extract_optimal_hyperparams(ab_results: Dict) -> Dict:
    clade_optimal_hyperparams = {}
    for clade, data in ab_results.items():
        if isinstance(data, list):
            test_accuracies = [i.testing_accuracy for i in data]
            best_fit_ix = test_accuracies.index(max(test_accuracies))
            clade_optimal_hyperparams[clade] = data[
                best_fit_ix
            ].hyperparameters
        elif isinstance(data, ResultsContainer):
            clade_optimal_hyperparams[clade] = data.hyperparameters
        else:
            raise TypeError(type(data))
    return clade_optimal_hyperparams


def plot_best_fits(
    results: Dict, optimal_hyperparams: Dict, filename: str = None
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
            best_model_results = next(
                filter(
                    lambda x: x.hyperparameters
                    == optimal_hyperparams[Ab][clade],
                    clade_data,
                )
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
        n += 1

    fig.text(
        0, 0.5, "Mean Accuracy per MIC Bin", va="center", rotation="vertical"
    )
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
) -> Dict[int, Union[Lasso, Ridge]]:

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
            model = fit_model(
                training_features, training_labels, model_type, **hps
            )
        models[clade] = model

    return models


# def evaluate_models(
#     Ab: str, models_dict: Dict[int, Union[Lasso, Ridge]], data_dir: str
# ):
#     adj = load_adjacency_matrix(data_dir)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    model_type = "ridge"
    results_dir = f"linear_model/results/{model_type}_results/gwas_filtered/cluster_wise_CV"  # noqa: E501

    results = get_results(results_dir)
    optimal_hyperparams = {
        Ab: extract_optimal_hyperparams(Ab_results)
        for Ab, Ab_results in results.items()
    }

    plot_best_fits(results, optimal_hyperparams)

    ab_best_models = {}
    for Ab, Ab_hps in optimal_hyperparams.items():
        root_dir = "data/gonno/model_inputs/freq_5_95/"
        data_dir = os.path.join(root_dir, f"log2_{Ab}_mic", "gwas_filtered")

        ab_best_models[Ab] = fit_Ab_models(Ab, Ab_hps, model_type, data_dir)

        # evaluate_models(Ab, models_dict, data_dir)
