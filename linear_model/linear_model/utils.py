import os
from functools import lru_cache
from dataclasses import dataclass
import nptyping
from typing import Dict, Any, Union, List

import torch
import pandas as pd
import numpy as np


@lru_cache(maxsize=1)
def load_training_data(data_dir):
    features = torch.load(os.path.join(data_dir, "training_features.pt"))
    labels = torch.load(os.path.join(data_dir, "training_labels.pt"))
    return features.to_dense(), labels


@lru_cache(maxsize=1)
def load_testing_data(data_dir):
    features = torch.load(os.path.join(data_dir, "testing_features.pt"))
    labels = torch.load(os.path.join(data_dir, "testing_labels.pt"))
    return features.to_dense(), labels


@lru_cache(maxsize=1)
def load_metadata(data_dir):
    training_metadata = pd.read_csv(os.path.join(data_dir, "training_metadata.csv"))
    testing_metadata = pd.read_csv(os.path.join(data_dir, "testing_metadata.csv"))
    return training_metadata, testing_metadata


def train_test_validate_split(
    training_data,
    testing_data,
    training_metadata,
    testing_metadata,
    left_out_clades: Union[List, int],
):
    if isinstance(left_out_clades, int):
        left_out_clades = [left_out_clades]

    training_indices = training_metadata.loc[
        training_metadata.clusters.isin(left_out_clades)
    ].index
    testing_indices = testing_metadata.loc[
        testing_metadata.clusters.isin(left_out_clades)
    ].index
    validation_indices_1 = training_metadata.loc[
        training_metadata.clusters.isin(left_out_clades)
    ].index  # extract data from training set
    validation_indices_2 = testing_metadata.loc[
        testing_metadata.clusters.isin(left_out_clades)
    ].index  # extract data from testing set

    training_features = torch.index_select(
        training_data[0], 0, torch.as_tensor(training_indices)
    )
    training_labels = torch.index_select(
        training_data[1], 0, torch.as_tensor(training_indices)
    )
    testing_features = torch.index_select(
        testing_data[0], 0, torch.as_tensor(testing_indices)
    )
    testing_labels = torch.index_select(
        testing_data[1], 0, torch.as_tensor(testing_indices)
    )
    validation_features = torch.cat(
        [
            torch.index_select(
                training_data[0], 0, torch.as_tensor(validation_indices_1)
            ),
            torch.index_select(
                testing_data[0], 0, torch.as_tensor(validation_indices_2)
            ),
        ]
    )
    validation_labels = torch.cat(
        [
            torch.index_select(
                training_data[1], 0, torch.as_tensor(validation_indices_1)
            ),
            torch.index_select(
                testing_data[1], 0, torch.as_tensor(validation_indices_2)
            ),
        ]
    )

    return (
        training_features,
        training_labels,
        testing_features,
        testing_labels,
        validation_features,
        validation_labels,
    )


def accuracy(
    predictions: nptyping.NDArray[nptyping.Float],
    labels: nptyping.NDArray[nptyping.Float],
) -> float:
    """
    Prediction accuracy defined as percentage of predictions within 1 twofold
    dilution of true value
    """
    diff = abs(predictions - labels)
    correct = diff[[i < 1 for i in diff]]
    return len(correct) / len(predictions) * 100


def mean_acc_per_bin(
    predictions: nptyping.NDArray[float], labels: nptyping.NDArray[float]
) -> float:
    """
    Splits labels into bins of size = bin_size, and calculates the prediction
    accuracy in each bin.
    Returns the mean accuracy across all bins
    """
    assert len(predictions) == len(labels) and len(labels) > 0

    # apply Freedman-Diaconis rule to get optimal bin size
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    IQR = np.subtract(*np.percentile(labels, [75, 25]))
    if IQR == 0:
        IQR = labels.max() - labels.min()
    bin_size = 2 * IQR / (len(labels) ** (1 / 3))
    bin_size = int(
        np.ceil(bin_size)
    )  # round up cause if less than 1 will not work with accuracy function

    min_value = int(np.floor(min(labels)))
    max_value = int(np.floor(max(labels)))
    bins = list(range(min_value, max_value + bin_size, bin_size))
    binned_labels = np.digitize(labels, bins)

    df = pd.DataFrame(
        {
            "labels": labels,
            "predictions": predictions,
            "binned_labels": binned_labels,
        }
    )  # to allow quick searches across bins

    # percentage accuracy per bin
    def _get_accuracy(d):
        acc = accuracy(
            d.labels.to_numpy(),
            d.predictions.to_numpy(),
        )
        return acc

    bin_accuracies = df.groupby(df.binned_labels).apply(_get_accuracy)

    return bin_accuracies.mean()


def load_adjacency_matrix(data_dir, degree_normalised=False):
    if degree_normalised:
        adj = torch.load(
            os.path.join(data_dir, "degree_normalised_unitig_adjacency_tensor.pt")
        )
    else:
        adj = torch.load(os.path.join(data_dir, "unitig_adjacency_tensor.pt"))
    adj = adj.coalesce()
    return adj


@lru_cache()
def convolve(features, adj):
    x = torch.sparse.mm(adj, features.transpose(0, 1))
    return x.transpose(0, 1)


@dataclass(unsafe_hash=True)
class ResultsContainer:
    training_accuracy: float
    testing_accuracy: float
    validation_accuracy: float

    training_MSE: float
    testing_MSE: float
    validation_MSE: float

    training_predictions: nptyping.NDArray[nptyping.Float]
    testing_predictions: nptyping.NDArray[nptyping.Float]
    validation_predictions: nptyping.NDArray[nptyping.Float]

    hyperparameters: Dict[str, float]

    model_type: str

    model: Any
