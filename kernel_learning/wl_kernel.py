#!/usr/bin/env python3

import grakel
import pickle
import logging
import os
import torch
import sys

from GNN_model.utils import (
    load_adjacency_matrix,
    load_training_data,
    load_testing_data,
    load_metadata,
)

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def train_test_validation_split(
    left_out_clade,
    training_features,
    testing_features,
    training_metadata,
    testing_metadata,
):
    logging.info(
        f"Formatting data for model with clade {left_out_clade} left out"
    )
    training_indices = training_metadata.loc[
        training_metadata.clusters != left_out_clade
    ].index
    testing_indices = testing_metadata.loc[
        testing_metadata.clusters != left_out_clade
    ].index
    validation_indices_1 = training_metadata.loc[
        training_metadata.clusters == left_out_clade
    ].index  # extract data from training set
    validation_indices_2 = testing_metadata.loc[
        testing_metadata.clusters == left_out_clade
    ].index  # extract data from testing set

    clade_training_features = torch.index_select(
        training_features,
        0,
        torch.as_tensor(training_indices, dtype=torch.int64),
    )
    clade_testing_features = torch.index_select(
        testing_features,
        0,
        torch.as_tensor(testing_indices, dtype=torch.int64),
    )
    validation_features = torch.cat(
        [
            torch.index_select(
                training_features,
                0,
                torch.as_tensor(validation_indices_1, dtype=torch.int64),
            ),
            torch.index_select(
                testing_features,
                0,
                torch.as_tensor(validation_indices_2, dtype=torch.int64),
            ),
        ]
    )

    return clade_training_features, clade_testing_features, validation_features


def format_data(data_dir, left_out_clade):
    adj = load_adjacency_matrix(data_dir, degree_normalised=False)
    adj = adj.coalesce()
    indices = adj.indices().tolist()

    edges = set(
        [(indices[0][i], indices[1][i]) for i in range(len(indices[0]))]
    )

    training_features = load_training_data(data_dir)[0]
    testing_features = load_testing_data(data_dir)[0]
    training_metadata, testing_metadata = load_metadata(data_dir)

    def parse_features(f):
        return {i: f[i] for i in range(len(f))}

    features = train_test_validation_split(
        left_out_clade,
        training_features,
        testing_features,
        training_metadata,
        testing_metadata,
    )

    G_train = [[edges, parse_features(i.tolist())] for i in features[0]]
    G_test = [[edges, parse_features(i.tolist())] for i in features[1]]
    G_validate = [[edges, parse_features(i.tolist())] for i in features[2]]

    return {"G_train": G_train, "G_test": G_test, "G_validate": G_validate}


def fit_wl_kernel(G_train):
    wl_kernel = grakel.kernels.weisfeiler_lehman.WeisfeilerLehman(
        verbose=True, normalize=False
    )
    k_train = wl_kernel.fit_transform(G_train)
    logging.info("Fitted WL kernel to training data and saved kernel")

    return wl_kernel, k_train


def transform_testing_data(wl_kernel, G_test):
    k_test = wl_kernel.transform(G_test)
    logging.info("Transformed testing data")

    return k_test


def save(out_dir, left_out_clade, G_train, G_test, G_validate):
    with open(
        os.path.join(
            out_dir,
            f"clade_{left_out_clade}_left_out_training_data_grakel_representation.pkl",
        ),
        "wb",
    ) as a:
        pickle.dump(G_train, a)
    with open(
        os.path.join(
            out_dir,
            f"clade_{left_out_clade}_left_out_testing_data_grakel_representation.pkl",
        ),
        "wb",
    ) as a:
        pickle.dump(G_test, a)
    with open(
        os.path.join(
            out_dir,
            f"clade_{left_out_clade}_left_out_validation_data_grakel_representation.pkl",
        ),
        "wb",
    ) as a:
        pickle.dump(G_validate, a)


def formatting_complete(out_dir, left_out_clade):
    formatted_files = []
    formatted_files.append(
        os.path.join(
            out_dir,
            f"clade_{left_out_clade}_left_out_training_data_grakel_representation.pkl",
        )
    )
    formatted_files.append(
        os.path.join(
            out_dir,
            f"clade_{left_out_clade}_left_out_testing_data_grakel_representation.pkl",
        )
    )
    formatted_files.append(
        os.path.join(
            out_dir,
            f"clade_{left_out_clade}_left_out_validation_data_grakel_representation.pkl",
        )
    )

    if all([os.path.isfile(i) for i in formatted_files]):
        return True
    else:
        return False


def load_formatted_data(out_dir, left_out_clade):
    with open(
        os.path.join(
            out_dir,
            f"clade_{left_out_clade}_left_out_training_data_grakel_representation.pkl",
        ),
        "rb",
    ) as a:
        G_train = pickle.load(a)
    with open(
        os.path.join(
            out_dir,
            f"clade_{left_out_clade}_left_out_testing_data_grakel_representation.pkl",
        ),
        "rb",
    ) as a:
        G_test = pickle.load(a)
    with open(
        os.path.join(
            out_dir,
            f"clade_{left_out_clade}_left_out_validation_data_grakel_representation.pkl",
        ),
        "rb",
    ) as a:
        G_validate = pickle.load(a)

    return G_train, G_test, G_validate


if __name__ == "__main__":
    Ab = sys.argv[1]
    left_out_clade = int(sys.argv[2])
    fit = True

    root_dir = "data/model_inputs/freq_5_95/"
    out_dir = f"kernel_learning/{Ab}/gwas_filtered/cluster_CV"

    pre_formatted = formatting_complete(out_dir, left_out_clade)

    if not pre_formatted:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        data_dir = os.path.join(root_dir, Ab, "gwas_filtered")
        data_by_left_out_cluster = format_data(data_dir, left_out_clade)

        G_train = data_by_left_out_cluster["G_train"]
        G_test = data_by_left_out_cluster["G_test"]
        G_validate = data_by_left_out_cluster["G_validate"]

        save(out_dir, left_out_clade, G_train, G_test, G_validate)

    if fit:
        if pre_formatted:
            G_train, G_test, G_validate = load_formatted_data(
                out_dir, left_out_clade
            )
        wl_kernel, k_train = fit_wl_kernel(G_train)
        with open(
            os.path.join(
                out_dir, f"clade_{left_out_clade}_left_out_k_train.pkl"
            ),
            "wb",
        ) as a:
            pickle.dump(k_train, a)
        with open(
            os.path.join(
                out_dir, f"clade_{left_out_clade}_left_out_wl_kernel.pkl"
            ),
            "wb",
        ) as a:
            pickle.dump(wl_kernel, a)

        k_test = transform_testing_data(wl_kernel, G_test)
        with open(
            os.path.join(
                out_dir, f"clade_{left_out_clade}_left_out_k_test.pkl"
            ),
            "wb",
        ) as a:
            pickle.dump(k_test, a)

        k_validate = transform_testing_data(wl_kernel, G_validate)
        with open(
            os.path.join(
                out_dir, f"clade_{left_out_clade}_left_out_k_validate.pkl"
            ),
            "wb",
        ) as a:
            pickle.dump(k_validate, a)