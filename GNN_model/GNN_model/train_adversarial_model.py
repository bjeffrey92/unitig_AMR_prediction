#!/usr/bin/env python3

import time
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim

from GNN_model.utils import (
    load_training_data,
    load_testing_data,
    load_labels_2,
    write_epoch_results,
    DataGenerator,
    logcosh,
    accuracy,
    country_accuracy,
    MetricAccumulator,
)
from GNN_model.adversarial_model import Adversary, MICPredictor
from tqdm import tqdm

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def load_data(data_dir, countries=True, families=False):
    training_features, training_labels = load_training_data(data_dir)
    testing_features, testing_labels = load_testing_data(data_dir)
    training_labels_2, testing_labels_2 = load_labels_2(data_dir, countries, families)
    assert (
        training_features.shape[1] == testing_features.shape[1]
    ), "Dimensions of training and testing data not equal"

    training_data = DataGenerator(training_features, training_labels, training_labels_2)
    testing_data = DataGenerator(testing_features, testing_labels, testing_labels_2)

    return training_data, testing_data


def epoch_(predictor, data, adversary=None):
    data.reset_generator()

    pred_outputs = [None] * data.n_samples
    adv_outputs = [None] * data.n_samples
    for i in tqdm(range(data.n_samples), desc="Epoch"):
        x = data.next_sample()[0]
        y, y_hat = predictor(x)
        pred_outputs[i] = y.unsqueeze(0)
        if adversary is None:
            continue
        z = adversary(y_hat)
        adv_outputs[i] = z.unsqueeze(0)

    if adversary is None:
        return torch.cat(pred_outputs, dim=0)
    else:
        return torch.cat(pred_outputs, dim=0), torch.cat(adv_outputs, dim=0)


def pre_train_predictor(
    predictor, pred_optimizer, pred_loss, training_data, epoch, testing_data=None
):
    t = time.time()
    predictor.train()
    pred_optimizer.zero_grad()

    training_data.reset_generator()
    training_data.shuffle_samples()

    pred_outputs = epoch_(predictor, training_data)
    print(pred_outputs, pred_outputs.var())

    loss_train = pred_loss(pred_outputs, training_data.labels)
    acc_train = accuracy(pred_outputs, training_data.labels)

    if testing_data:
        loss_test, acc_test = test(testing_data, pred_loss, accuracy, predictor)
    else:
        loss_test = "N/A"
        acc_test = "N/A"

    loss_train.backward()
    pred_optimizer.step()
    loss_train = float(loss_train)  # to write to file

    logging.info(
        "Predictor pretraining:\n"
        + f"Epoch {epoch} complete\n"
        + f"\tTime taken = {time.time() - t}\n"
        + f"\tTraining Data Loss = {loss_train}\n"
        + f"\tTraining Data Accuracy = {acc_train}\n"
        f"\tTesting Data Loss = {loss_test}\n"
        + f"\tTesting Data Accuracy = {acc_test}\n"
    )

    return predictor, (loss_train, acc_train, loss_test, acc_test)


def pre_train_adversary(
    predictor,
    adversary,
    adv_optimizer,
    adv_loss,
    training_data,
    epoch,
    testing_data=None,
):

    t = time.time()
    predictor.train(False)  # used to create input to adversary
    adversary.train()
    adv_optimizer.zero_grad()

    training_data.reset_generator()
    training_data.shuffle_samples()

    adv_outputs = epoch_(predictor, training_data, adversary)[1]

    loss_train = adv_loss(adv_outputs, training_data.labels_2)
    acc_train = country_accuracy(adv_outputs, training_data.labels_2)

    if testing_data:
        loss_test, acc_test = test(
            testing_data, adv_loss, country_accuracy, predictor, adversary
        )
    else:
        loss_test = "N/A"
        acc_test = "N/A"

    loss_train.backward()
    adv_optimizer.step()
    loss_train = float(loss_train)

    logging.info(
        "Adversary pretraining:\n"
        + f"Epoch {epoch} complete\n"
        + f"\tTime taken = {time.time() - t}\n"
        + f"\tAdversary Training Loss = {loss_train}\n"
        + f"\tAdversary Training Accuracy = {acc_train}\n"
        f"\tAdversary Testing Loss = {loss_test}\n"
        + f"\tAdversary Testing Acc = {acc_test}\n"
    )

    return adversary, (loss_train, acc_train, loss_test, acc_test)


def adversarial_training(
    predictor,
    pred_optimizer,
    pred_loss,
    training_data,
    adj,
    epoch,
    adversary,
    adv_loss,
    adv_optimizer,
    lbda,
    testing_data=None,
):

    t = time.time()
    predictor.train()
    adversary.train()
    pred_optimizer.zero_grad()
    adv_optimizer.zero_grad()

    training_data.shuffle_samples()

    # train predictor
    for param in predictor.parameters():
        param.requires_grad = True
    for param in adversary.parameters():
        param.requires_grad = False
    pred_outputs, adv_outputs = epoch_(predictor, training_data, adversary)

    loss_adversary = adv_loss(adv_outputs, training_data.labels_2)
    loss_pred = pred_loss(pred_outputs, training_data.labels)
    loss_pred = (
        loss_pred - lbda * loss_adversary
    )  # adjusted loss function for predictor
    loss_pred.backward()
    pred_optimizer.step()

    # train adversary
    for param in predictor.parameters():
        param.requires_grad = False
    for param in adversary.parameters():
        param.requires_grad = True
    pred_outputs, adv_outputs = epoch_(predictor, training_data, adversary)

    loss_adversary = adv_loss(adv_outputs, training_data.labels_2)
    loss_adversary.backward()
    adv_optimizer.step()

    # for logging
    loss_adversary = float(loss_adversary)
    acc_adversary = country_accuracy(adv_outputs, training_data.labels_2)
    for param in predictor.parameters():
        param.requires_grad = True
    predictor.train(False)
    loss_train = float(pred_loss(pred_outputs, training_data.labels))
    acc_train = accuracy(pred_outputs, training_data.labels)

    if testing_data:
        loss_test, acc_test = test(testing_data, adj, pred_loss, accuracy, predictor)
    else:
        loss_test = "N/A"
        acc_test = "N/A"

    logging.info(
        "Adversarial Training:\n"
        + f"Epoch {epoch} complete\n"
        + f"\tTime taken = {time.time() - t}\n"
        + f"\tTraining Data Loss = {loss_train}\n"
        + f"\tTraining Data Accuracy = {acc_train}\n"
        f"\tTesting Data Loss = {loss_test}\n"
        + f"\tTesting Data Accuracy = {acc_test}\n"
        + f"\tAdversary Training Data Loss = {loss_adversary}\n"
        + f"\tAdversary Training Data Accuracy = {acc_adversary}\n"
    )

    return predictor, adversary, (loss_train, acc_train, loss_test, acc_test)


def test(data, loss_function, accuracy, predictor, adversary=None):
    data.reset_generator()
    predictor.train(False)

    if adversary is not None:
        adversary.train(False)
        output = epoch_(predictor, data, adversary)
        loss = float(loss_function(output[1], data.labels_2))
        acc = accuracy(output[1], data.labels_2)
    else:
        output = epoch_(predictor, data, adversary)
        loss = float(loss_function(output, data.labels))
        acc = accuracy(output, data.labels)

    return loss, acc


def main(Ab: str):
    data_dir = os.path.join(root_dir, Ab)

    (
        training_data,
        testing_data,
    ) = load_data(data_dir, countries=False, families=True)

    predictor = MICPredictor(
        n_feat=training_data.n_nodes, n_hid_1=100, n_hid_2=100, out_dim=1, dropout=0.3
    )
    predictor.initialise_weights_and_biases(0)
    adversary = Adversary(
        n_feat=100,
        n_hid_1=50,
        n_hid_2=50,
        dropout=0.3,
        out_dim=max(training_data.labels_2.tolist()) + 1,
    )
    adversary.initialise_weights_and_biases(0)

    pred_optimizer = optim.Adam(predictor.parameters(), lr=0.001, weight_decay=5e-3)
    pred_loss = logcosh
    adv_optimizer = optim.Adam(adversary.parameters(), lr=0.001, weight_decay=5e-3)
    adv_loss = nn.CrossEntropyLoss()

    # pretraining predictor
    pred_training_metrics = MetricAccumulator()
    for param in predictor.parameters():
        param.requires_grad = True
    for epoch in range(60):
        epoch += 1
        predictor, epoch_results = pre_train_predictor(
            predictor,
            pred_optimizer,
            pred_loss,
            training_data,
            epoch,
            testing_data,
        )
        pred_training_metrics.add(epoch_results)
        pred_training_metrics.log_gradients(epoch)
        write_epoch_results(epoch, epoch_results, f"{Ab}_predictor_pretraining.tsv")

    torch.save(predictor, f"{Ab}_pretrained_predictor.pt")

    # pretraining adversary
    adv_training_metrics = MetricAccumulator()
    for param in predictor.parameters():
        param.requires_grad = False  # gradient only calculated over adversary network
    for epoch in range(500):
        epoch += 1
        adversary, epoch_results = pre_train_adversary(
            predictor,
            adversary,
            adv_optimizer,
            adv_loss,
            training_data,
            epoch,
            testing_data,
        )
        adv_training_metrics.add(epoch_results)
        adv_training_metrics.log_gradients(epoch)
        write_epoch_results(epoch, epoch_results, f"{Ab}_adversary_pretraining.tsv")

        if (
            len(
                [
                    i
                    for i in adv_training_metrics.testing_data_acc_grads[-10:]
                    if i < 0.1
                ]
            )
            >= 10
            and epoch > 50
        ):
            logging.info(
                "Gradient of testing data accuracy appears to have plateaued, terminating early"
            )
            break

    torch.save(adversary, f"{Ab}_pretrained_adversary.pt")

    # Adversarial training
    loss_pred = int(
        pred_training_metrics.training_data_loss[-1]
    )  # final loss value of each model used to guess appropriate value for lbda
    loss_adv = int(adv_training_metrics.training_data_loss[-1])

    lbda = loss_pred / loss_adv  # weighting between adversary and predictor loss

    training_metrics = MetricAccumulator()
    for epoch in range(500):
        epoch += 1
        predictor, adversary, epoch_results = adversarial_training(
            predictor,
            pred_optimizer,
            pred_loss,
            training_data,
            epoch,
            adversary,
            adv_loss,
            adv_optimizer,
            lbda,
            testing_data,
        )
        training_metrics.add(epoch_results)
        training_metrics.log_gradients(epoch)
        write_epoch_results(
            epoch, epoch_results, f"{Ab}_adversarial_predictor_training.tsv"
        )

        # less stringent plateauing criteria for adversarial training
        if (
            len([i for i in training_metrics.testing_data_acc_grads[-30:] if i < 0.1])
            >= 10
            and epoch > 50
        ):
            logging.info(
                "Gradient of testing data accuracy appears to have plateaued, terminating early"
            )
            break

    torch.save(predictor, f"{Ab}_fitted_predictor.pt")
    torch.save(adversary, f"{Ab}_fitted_adversary.pt")


if __name__ == "__main__":
    root_dir = "data/gonno/model_inputs/family_normalised"
    Abs = ["log2_azm_mic", "log2_cip_mic", "log2_cro_mic", "log2_cfx_mic"]
    for Ab in Abs:
        main(Ab)
