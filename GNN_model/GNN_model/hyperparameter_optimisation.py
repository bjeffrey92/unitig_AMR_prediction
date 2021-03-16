#!/usr/bin/env python3

import sys
import pickle
import os
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from math import log10
from bayes_opt import BayesianOptimization

from GNN_model import train, utils, models


logging.basicConfig()
logging.root.setLevel(logging.INFO)


def train_evaluate(
    Ab,
    left_out_cluster,
    dropout,
    l2_alpha,
    lr,
    n_hid_1,
    n_hid_2,
    verbose=False,
):

    data_dir = os.path.join("data/model_inputs/freq_5_95", Ab, "gwas_filtered")

    inputs = train.load_data(
        data_dir, distances=False, adj=True, left_out_cluster=left_out_cluster
    )
    training_data, testing_data, validation_data, adj = inputs

    # these hyperparams are selected from log uniform distribution
    l2_alpha = 10 ** l2_alpha
    lr = 10 ** lr

    torch.manual_seed(0)
    model = models.GCNPerNode(
        adj.shape[0], int(n_hid_1), int(n_hid_2), out_dim=1, dropout=dropout
    )

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=l2_alpha
    )  # weight decay is l2 loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=verbose
    )
    loss_function = nn.MSELoss()

    training_metrics = utils.MetricAccumulator()
    for epoch in range(40):
        epoch += 1

        model, epoch_results = train.train(
            training_data,
            model,
            optimizer,
            epoch,
            loss_function,
            utils.mean_acc_per_bin,
            adj,
            testing_data=testing_data,
            validation_data=validation_data,
            verbose=verbose,
        )
        training_metrics.add(epoch_results)

        # will adjust lr when loss plateaus
        scheduler.step(training_metrics.training_data_loss[-1])

        # if testing data accuracy has plateaued
        if (
            len(
                [
                    i
                    for i in training_metrics.testing_data_acc_grads[-10:]
                    if i < 0.1
                ]
            )
            >= 10
            and epoch > 10
        ):
            break

    # returns average testing data loss in the last five epochs
    return (
        -sum(training_metrics.testing_data_loss[-5:]) / 5
    )  # negative so can use maximization function


def predict(data, model, accuracy, adj=None):
    data.reset_generator()
    model.train(False)

    with torch.no_grad():
        output = train.epoch_(model, data, adj)
    acc = accuracy(output, data.labels)

    return output, acc


def fit_best_model(
    Ab, left_out_cluster, dropout, l2_alpha, lr, n_hid_1, n_hid_2
):

    data_dir = os.path.join("data/model_inputs/freq_5_95", Ab)

    inputs = train.load_data(
        data_dir, distances=False, adj=True, left_out_cluster=left_out_cluster
    )
    training_data, testing_data, validation_data, adj = inputs

    torch.manual_seed(0)
    model = models.GCNPerNode(
        adj.shape[0], int(n_hid_1), int(n_hid_2), out_dim=1, dropout=dropout
    )

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=l2_alpha
    )  # weight decay is l2 loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )
    loss_function = nn.MSELoss()

    training_metrics = utils.MetricAccumulator()
    for epoch in range(200):
        epoch += 1

        model, epoch_results = train.train(
            training_data,
            model,
            optimizer,
            epoch,
            loss_function,
            utils.mean_acc_per_bin,
            adj,
            testing_data=testing_data,
            validation_data=validation_data,
            verbose=True,
        )
        training_metrics.add(epoch_results)

        # will adjust lr when loss plateaus
        scheduler.step(training_metrics.training_data_loss[-1])

        if epoch >= 20:
            training_metrics.log_gradients(epoch)

        if (
            len(
                [
                    i
                    for i in training_metrics.testing_data_acc_grads[-10:]
                    if i < 0.1
                ]
            )
            >= 10
            and epoch > 50
        ):
            break

    train_pred, train_acc = predict(
        training_data, model, utils.mean_acc_per_bin, adj
    )
    test_pred, test_acc = predict(
        testing_data, model, utils.mean_acc_per_bin, adj
    )
    validation_pred, validation_acc = predict(
        validation_data, model, utils.mean_acc_per_bin, adj
    )

    accuracies = (
        {
            "training_accuracy": train_acc,
            "testing_accuracy": test_acc,
            "validation_accuracy": validation_acc,
        },
    )
    model_outputs = {
        "training_predictions": train_pred,
        "training_labels": training_data.labels,
        "testing_predictions": test_pred,
        "testing_labels": testing_data.labels,
        "validation_predictions": validation_pred,
        "validation_labels": validation_data.labels,
    }

    return accuracies, model_outputs


def save_CV_results(accuracies, out_dir, fname):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, fname)
    with open(out_file, "wb") as a:
        pickle.dump(accuracies, a)


if __name__ == "__main__":

    Ab = sys.argv[1]

    pbounds = {
        "dropout": (0.2, 0.5),
        "l2_alpha": (log10(1e-4), log10(1e-2)),
        "lr": (log10(1e-4), log10(1e-2)),
        "n_hid_1": (50, 200),
        "n_hid_2": (10, 50),
    }

    clade_wise_results = {}
    for left_out_cluster in list(range(5, 7)):
        logging.info(f"Ab = {Ab}, left out cluster = {left_out_cluster}")

        partial_fitting_function = partial(
            train_evaluate, Ab=Ab, left_out_cluster=left_out_cluster
        )

        optimizer = BayesianOptimization(
            f=partial_fitting_function, pbounds=pbounds, random_state=1
        )

        optimizer.maximize(n_iter=20)
        logging.info(
            f"Completed hyperparam optimisation: {Ab}, {left_out_cluster}"
        )

        best_hyperparams = {}
        best_hyperparams["dropout"] = optimizer.max["params"]["dropout"]
        best_hyperparams["l2_alpha"] = (
            10 ** optimizer.max["params"]["l2_alpha"]
        )
        best_hyperparams["lr"] = 10 ** optimizer.max["params"]["lr"]
        best_hyperparams["n_hid_1"] = int(optimizer.max["params"]["n_hid_1"])
        best_hyperparams["n_hid_2"] = int(optimizer.max["params"]["n_hid_2"])

        accuracies, model_outputs = fit_best_model(
            Ab,
            left_out_cluster,
            dropout=best_hyperparams["dropout"],
            l2_alpha=best_hyperparams["l2_alpha"],
            lr=best_hyperparams["lr"],
            n_hid_1=best_hyperparams["n_hid_1"],
            n_hid_2=best_hyperparams["n_hid_2"],
        )
        logging.info(
            f"Fitted model with best hyperparams: {Ab}, {left_out_cluster}"
        )

        clade_wise_results[left_out_cluster] = {
            "params": best_hyperparams,
            "accuracies": accuracies,
            "model_outputs": model_outputs,
        }

        # saves to file after each clade
        out_dir = "GNN_model/Results/mean_acc_per_bin/Cluster_wise_CV"
        fname = f"{Ab}_GCNPerNode3.pkl"
        save_CV_results(clade_wise_results, out_dir, fname)
