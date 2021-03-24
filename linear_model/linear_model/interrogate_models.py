import os
import pickle
import re
import glob
import warnings
import logging
from typing import Dict

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.exceptions import ConvergenceWarning

from linear_model.utils import (
    load_training_data,
    load_testing_data,
    load_metadata,
    train_test_validate_split,
    ResultsContainer,
)

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def get_results(results_dir: str) -> Dict:
    files = glob.glob(results_dir + "/*pkl")
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


if __name__ == "__main__":
    results_dir = (
        "linear_model/results/ridge_results/gwas_filtered/cluster_wise_CV"
    )
    model = "ridge"

    results = get_results(results_dir)
    model_hyperparams = {
        Ab: extract_optimal_hyperparams(results)
        for Ab, results in results.items()
    }
