#!/usr/bin/env python3

import time
import logging
import math

import torch
import torch.nn as nn
import torch.optim as optim

from GNN_model.utils import load_training_data, load_testing_data, \
                        load_adjacency_matrix, load_labels_2, save_model, \
                        write_epoch_results, DataGenerator, logcosh, \
                        accuracy, country_accuracy, MetricAccumulator
from GNN_model.adversarial_model import MICPredictor, Adversary

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def load_data(data_dir, countries = True, families = False):
    adj = load_adjacency_matrix(data_dir)
    training_features, training_labels = load_training_data(data_dir)
    testing_features, testing_labels = load_testing_data(data_dir)
    training_labels_2, testing_labels_2 = load_labels_2(data_dir, 
                                                        countries, families)
    assert training_features.shape[1] == testing_features.shape[1], \
        'Dimensions of training and testing data not equal'
    
    training_data = DataGenerator(training_features, training_labels, 
                                training_labels_2)
    testing_data = DataGenerator(testing_features, testing_labels, 
                                testing_labels_2)

    return training_data, testing_data, adj


def epoch_(data, adj, predictor, adversary = None):
    data.reset_generator()

    pred_outputs = [None] * data.n_samples
    adv_outputs = [None] * data.n_samples
    for i in range(data.n_samples):
        x = data.next_sample()[0]
        y, y_hat = predictor(x, adj)
        pred_outputs[i] = y.unsqueeze(0)
        if adversary is None: continue
        z = adversary(y_hat)
        adv_outputs[i] = z.unsqueeze(0)
        
    if adversary is None:
        return torch.cat(pred_outputs, dim = 0)
    else:
        return torch.cat(pred_outputs, dim = 0), torch.cat(adv_outputs, dim = 0)
    

def pre_train_predictor(predictor, pred_optimizer, pred_loss, 
                training_data, adj, epoch, testing_data = None):
    t = time.time()
    predictor.train()
    pred_optimizer.zero_grad()

    training_data.reset_generator()
    training_data.shuffle_samples()

    pred_outputs = epoch_(training_data, adj, predictor)

    loss_train = pred_loss(pred_outputs, training_data.labels)
    acc_train = accuracy(pred_outputs, training_data.labels)

    if testing_data:
        loss_test, acc_test = test(testing_data, adj, pred_loss, 
                                accuracy, predictor)
    else:
        loss_test = 'N/A'
        acc_test = 'N/A'

    loss_train.backward()
    pred_optimizer.step()
    loss_train = float(loss_train) #to write to file

    logging.info('Predictor pretraining:\n' + \
            f'Epoch {epoch} complete\n' + \
            f'\tTime taken = {time.time() - t}\n' + \
            f'\tTraining Data Loss = {loss_train}\n' + \
            f'\tTraining Data Accuracy = {acc_train}\n'
            f'\tTesting Data Loss = {loss_test}\n' + \
            f'\tTesting Data Accuracy = {acc_test}'
            )

    return predictor, (loss_train, acc_train, loss_test, acc_test)


def pre_train_adversary(predictor, adversary, adv_optimizer, adv_loss, 
                    training_data, adj, epoch, testing_data = None):
    
    t = time.time()
    predictor.train(False) #used to create input to adversary
    adversary.train()
    adv_optimizer.zero_grad()

    training_data.reset_generator()
    training_data.shuffle_samples()
    
    adv_outputs = epoch_(training_data, adj, predictor, adversary)[1]
    
    loss_train = adv_loss(adv_outputs, training_data.labels_2)
    acc_train = country_accuracy(adv_outputs, training_data.labels_2)

    if testing_data:
        loss_test, acc_test = test(testing_data, adj, adv_loss, 
                                country_accuracy, predictor, adversary)
    else:
        loss_test = 'N/A'
        acc_test = 'N/A'
    
    loss_train.backward()
    adv_optimizer.step()
    loss_train = float(loss_train)

    logging.info('Adversary pretraining:\n' + \
                f'Epoch {epoch} complete\n' + \
                f'\tTime taken = {time.time() - t}\n' + \
                f'\tPredictor Training Loss = {loss_train}\n' + \
                f'\tPredictor Training Accuracy = {acc_train}\n'
                f'\tAdversary Training Loss = {loss_train}\n' + \
                f'\tAdversary Training Acc = {acc_train}'
                )

    return adversary, (loss_train, acc_train, loss_train, acc_train)


def test(data, adj, loss_function, accuracy, predictor, adversary = None):
    data.reset_generator()
    predictor.train(False)
    
    output = epoch_(data, adj, predictor, adversary)
    if adversary is not None:
        adversary.train(False)
        loss = float(loss_function(output[1], data.labels_2))
        acc = accuracy(output[1], data.labels_2)
    else:
        loss = float(loss_function(output, data.labels))
        acc = accuracy(output, data.labels)

    return loss, acc


def main():
    data_dir =  'data/model_inputs/country_normalised/log2_azm_mic/'

    training_data, testing_data, adj = load_data(data_dir, 
                                                countries = True, 
                                                families = False)

    predictor = MICPredictor(n_feat = training_data.n_nodes,
                            n_hid_1 = 50, 
                            n_hid_2 = 50,
                            n_hid_3 = 20,
                            out_dim = 1,
                            dropout = 0.5)    
    adversary = Adversary(n_feat = 20,
                        n_hid = 20, 
                        out_dim = max(training_data.labels_2.tolist()) + 1)
    
    pred_optimizer = optim.Adam(predictor.parameters(), lr = 0.0001,
                                weight_decay = 5e-3)
    pred_loss = logcosh
    adv_optimizer = optim.Adam(adversary.parameters(), lr = 0.0001, 
                        weight_decay = 5e-4)
    adv_loss = nn.CrossEntropyLoss()

    #pretraining predictor
    training_metrics = MetricAccumulator()
    for param in predictor.parameters():
        param.requires_grad = True
    for epoch in range(60):
        epoch += 1
        predictor, epoch_results = pre_train_predictor(predictor, 
                                                        pred_optimizer, 
                                                        pred_loss, 
                                                        training_data, 
                                                        adj, 
                                                        epoch, 
                                                        testing_data)
        training_metrics.add(epoch_results)
        training_metrics.log_gradients(epoch)

    #pretraining adversary
    training_metrics = MetricAccumulator()
    for param in predictor.parameters():
        param.requires_grad = False #gradient only calculated over adversary network
    for epoch in range(60):
        epoch += 1
        adversary, epoch_results = pre_train_adversary(predictor, 
                                                    adversary, 
                                                    adv_optimizer, 
                                                    adv_loss, 
                                                    training_data, 
                                                    adj, 
                                                    epoch, 
                                                    testing_data = None)
        training_metrics.add(epoch_results)
        training_metrics.log_gradients(epoch)


# if __name__ == '__main__':
#     main()